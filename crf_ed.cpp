#include "crf_ed.h"
#include <math.h>
#include <vector>
#include <cmath>
#include <set>
#include <unistd.h>

using std::vector;
using std::cerr;
using namespace std;

const double NINF = log(0);

//fill forward table to evaluate params on current example
void Crf_Ed::evaluate(int y,double & z0, double & z1){  
    table_A.reset();
    fill_table_A();
    z0=(table_A.get(words->I-1,words->J-1,end_state_match));
    z1=(table_A.get(words->I-1,words->J-1,end_state_mismatch));
    double total;
    total=logsumexp(z0,z1);
    z0=exp(z0-total);
    z1=exp(z1-total);
}
    
//the sum of x and y in the log-domain
double Crf_Ed::logsumexp(double x,double y){
    double z=max(x,y);
    if (isinf(z)){
        return NINF;
    }
    return log(exp(x-z)+exp(y-z))+z;
}

//run forward-backward algorithm
void Crf_Ed::fill_tables(){
    table_A.reset();
    table_B.reset();
    table_C.reset();
    table_D.reset();
    fill_table_A();
    fill_table_B();
    fill_table_C();
    fill_table_D();
}

//set the current example pair to datapoint
void Crf_Ed::set_words(Data_Element* datapoint){
    int I=datapoint->I ;
    int maxI=table_A.get_max_I() ;
    int J=datapoint->J ;
    int maxJ=table_A.get_max_J() ;
    if (I<maxI && J<maxJ){
        words=datapoint;
    }else{
        cerr<<"Datapoint too large for tables (I,J) = ("<<I<<","<<J<<") and \
            table.(I,J) = ("<<maxI<<","<<maxJ<<")"<<endl;
    }
}

//1 if the current example pair is correctly classified, 0 otherwise
double Crf_Ed::get_accuracy(int y){
    double z0=(table_A.get(words->I-1,words->J-1,end_state_match));
    double z1=(table_A.get(words->I-1,words->J-1,end_state_mismatch));
    double total=logsumexp(z0,z1);
    z0=exp(z0-total);
    z1=exp(z1-total);
    if (y==0){
        return 1*(z0>=z1);
    }else if (y==1){
        return 1*(z0<z1);
    }
    return 0;
}

//run only the forward algorithm to find the likelihood of the current example
void Crf_Ed::fill_table_ll(){
    table_A.reset();
    fill_table_A();
}

//work out the current point's log-likelihood after fill_table_ll() has run
double Crf_Ed::get_ll(int y){
    double z0=(table_A.get(words->I-1,words->J-1,end_state_match));
    double z1=(table_A.get(words->I-1,words->J-1,end_state_mismatch));
    double total=logsumexp(z0,z1);
    z0=z0-total;
    z1=z1-total;
    if (y==0){
        return (z0);
    }else if (y==1){
        return (z1);
    }
    return 1;//shouldn't happen
}

//helper to determine whether state is part of the label = 1 or label = 0 part
//of the state machine.
bool Crf_Ed::right_transducer(int z,int state){
    if(z==0){
        if((state>=1 && state<4) || state==end_state_match){
            return true;
        }
    }
    if(z==1){
        if((state>=4 && state<7 )|| state==end_state_mismatch){
            return true;
        }
    }
    return false;
}

//work out derivative of parameters for current example and 
//add it to vector derivs
void Crf_Ed::get_derivs(int z,vector<double> & derivs){ 
    vector<pair<double,double> > Ef (derivs.size());
    for (int i=0;i<(int)Ef.size();i++){
        Ef[i]=pair<double,double>(NINF,NINF);
    }
    double norm=NINF;
    double norm_z=NINF;
    norm=logsumexp(norm,table_A.get(words->I-1,words->J-1,end_state_match));
    norm=logsumexp(norm,table_A.get(words->I-1,words->J-1,end_state_mismatch));
    if (z==0){
        norm_z=logsumexp(norm_z,table_A.get(words->I-1,
                                            words->J-1,end_state_match));
    }else{
        norm_z=logsumexp(norm_z,table_A.get(words->I-1,
                                            words->J-1,end_state_mismatch));
    }

    for (int i=0;i<words->I;i++){
        for(int j=0;j<words->J;j++){
            //feature functions that are functions of current and previous
            //states
            for (int prevs=0;prevs<S;prevs++){
                for (int s=0;s<S;s++){
                    for (int e=0;e<E;e++){
                        int di=transitions[e].first;
                        int dj=transitions[e].second;
                        if (i+di>=0 && j+dj>=0){
                            double E_f=(table_C.get(i,j,prevs,s));
                            double E_ef=NINF;
                            if(right_transducer(z,s)){
                                E_ef=(table_C.get(i,j,prevs,s));//-E_ef_normal;
                            }
                            get_features_ij(i,j,prevs,s,e);
                            for(int f=0;f<vij_length;f++){
                            Ef[vij[f]].first=logsumexp(E_ef,Ef[vij[f]].first);
                            Ef[vij[f]].second=logsumexp(E_f,Ef[vij[f]].second);
                            }
                        }
                    }
                }
            }
            //feature functions that are functions of only current state
            for (int s=0;s<S;s++){
                for (int e=0;e<E;e++){
                    int di=transitions[e].first;
                    int dj=transitions[e].second;
                    if (i+di>=0 && j+dj>=0){
                        double E_f=(table_D.get(i,j,s));
                        double E_ef=NINF;
                        if(right_transducer(z,s)){
                            E_ef=(table_D.get(i,j,s));
                        }
                        get_features_j(i,j,s,e);
                        for(int f=0;f<vj_length;f++){
                            Ef[vj[f]].first=logsumexp(E_ef,Ef[vj[f]].first);
                            Ef[vj[f]].second=logsumexp(E_f,Ef[vj[f]].second);
                        }
                    }
                }
            }
        }//j
    }//i
    for (int f=0;f<(int)Ef.size();f++){
        derivs[f]+=1/3.0*(exp(Ef[f].first-norm_z) - exp(Ef[f].second-norm));
    }
}

//initialize the parameter vector by setting the weights representing
//disallowed transitions to -INF
void Crf_Ed::init_params(){
    for(int s=0;s<S;s++){
        for (int e=0;e<E;e++){
            double v=0;
            if ((s-1)%3 != e){
                v=NINF;
            }
            if (s==0 || s==end_state_match || s== end_state_mismatch){
                v=0.0;
            }
            params.push_back(v);
        }
    }
    for (int o=0;o<O;o++){
        for (int s=0;s<S;s++){
            double v=0.0;
            if (s==0 && o==1){
                v=NINF;
            }
            params.push_back(v);
        }
        for (int si=0;si<S;si++){
            for (int sj=0;sj<S;sj++){
                double v=0.0;
                if ((si>=1 && si<4 && sj>=4 && sj<7) || 
                    (sj>=1 && sj<4 && si>=4 && si<7)){
                    v=NINF;
                }
                if ((si>=1 && si<4 && sj==end_state_mismatch) || 
                    (si>=4 && si<7 && sj==end_state_match)){
                    v=NINF;
                }
                if (sj==0){
                    v=NINF;
                }
                if (si==end_state_match || si ==end_state_mismatch){
                    v=NINF;
                }
                if (si==0 && (sj==end_state_match || sj==end_state_mismatch)){
                    v=NINF;
                }
                if (si==0 && (sj!=1 &&sj!=4)){
                    v=NINF;
                }
                if ((sj==end_state_match || sj==end_state_mismatch) && o==2){
                    v=NINF;
                }
                params.push_back(v);
            }
        }
    }
}


//reference to parametervector    
vector<double>* Crf_Ed::get_params(){
    return &params;
}

//constructor
Crf_Ed::Crf_Ed(int MAX_I,int MAX_J,int MAX_S,int MAX_OBSERVATIONS) :
    table_A(NINF,MAX_I,MAX_J,MAX_S,1),
    table_B(NINF,MAX_I,MAX_J,MAX_S,1),
    table_C(NINF,MAX_I,MAX_J,MAX_S,MAX_S),
    table_D(NINF,MAX_I,MAX_J,MAX_S,1),
    vj(MAX_OBSERVATIONS,-1),
    vij(MAX_OBSERVATIONS,-1)
{
    S=MAX_S;
    O=MAX_OBSERVATIONS;
    end_state_match=7;
    end_state_mismatch=8;

    transitions.push_back(pair<int,int>(-1,-1));
    transitions.push_back(pair<int,int>(0,-1));
    transitions.push_back(pair<int,int>(-1,0));
    E=transitions.size();
    
    init_params();
}

//set class member vj to list of parameters that are activated at the
//current position i,j in the lattice for state s and transition e
//(NOTE: changed to global vj and vj_length for performance)
void Crf_Ed::get_features_j(int i,int j,int s,int e){
    int a=0;
    vj[a]=(e+E*s);
    a++;
    vector<int> * features=words->get_features(i,j);
    int F=(int)features->size();
    for (int o=0;o<F;o++){
        vj[a]=(E*S + s + (S+S*S)*features->at(o));
        a++;
    }
    vj_length=a;
}

//return sum of parameters returned by get_features_j()
double Crf_Ed::get_feature_sum_j(int i,int j,int s,int e){
    double total=0.0;
    get_features_j(i,j,s,e);
    for (int p=0;p<vj_length;p++){
        total+=params[vj[p]];
    }
    return total;
}

//similar to get_features_j but also include features activated by a 
//combination of current state s and previous state ps
void Crf_Ed::get_features_ij(int i,int j,int ps,int s,int e){
    int a=0;
    vector<int> * features=words->get_features(i,j);
    int F=(int)features->size();
    for (int o=0;o<F;o++){
        vij[a]=(E*S + S + s+ps*S + (S+S*S)*features->at(o));
        a++;
    }
    vij_length=a;
}

//return sum of parameters of get_features_ij
double Crf_Ed::get_feature_sum_ij(int i,int j,int ps,int s,int e){
    double total=0.0;
    get_features_ij(i,j,ps,s,e);
    for (int p=0;p<vij_length;p++){
        total+=params[vij[p]];
    }
    return total;
}

//private:

//fill forward table
void Crf_Ed::fill_table_A(){
    table_A.set(0.0,0,0,0); //for position i=0,j=0 and state=0, value is log(1)
    for (int i=0;i<words->I;i++){
        for (int j=0;j<words->J;j++){
            for (int prevs=0;prevs<S;prevs++){
                for (int s=0;s<S;s++){
                    for (int e=0;e<E;e++){
                        int di=transitions[e].first;
                        int dj=transitions[e].second;
                        if (i+di>=0 && j+dj>=0){
                            double lambsum=get_feature_sum_ij(i,j,prevs,s,e);
                            lambsum += get_feature_sum_j(i,j,s,e);
                            double newvalue=logsumexp(table_A.get(i+di,j+dj,
                                        prevs)+lambsum, table_A.get(i,j,s));
                            table_A.set(newvalue,i,j,s);
                        }
                    }
                }
            }
        }
    }
}

//fill backward table 
void Crf_Ed::fill_table_B(){
    table_B.set(0.0,words->I-1,words->J-1,end_state_match);
    table_B.set(0.0,words->I-1,words->J-1,end_state_mismatch);
    for (int i=words->I-1;i>=0;i--){
        for (int j=words->J-1;j>=0;j--){
            for (int prevs=0;prevs<S;prevs++){
                for (int s=0;s<S;s++){
                    for (int e=0;e<E;e++){
                        int di=transitions[e].first;
                        int dj=transitions[e].second;
                        if (i-di<=words->I-1 && j-dj<=words->J-1){
                            double lambsum=get_feature_sum_ij(i-di,j-dj,
                                                              prevs,s,e);
                            lambsum += get_feature_sum_j(i-di,j-dj,s,e);
                            double newvalue=logsumexp(table_B.get(i-di,j-dj,s)
                                    +lambsum, table_B.get(i,j,prevs));
                            table_B.set(newvalue,i,j,prevs);
                        }
                    }
                }
            }
        }
    }
}   

//combine forward and backward tables to find marginals of pairs of variables
void Crf_Ed::fill_table_C(){
    for (int i=0;i<words->I;i++){
        for (int j=0;j<words->J;j++){
            for (int prevs=0;prevs<S;prevs++){
                for (int s=0;s<S;s++){
                    for (int e=0;e<E;e++){
                        int di=transitions[e].first;
                        int dj=transitions[e].second;
                        if (i+di>=0 && j+dj>=0){
                            double lambsum=get_feature_sum_ij(i,j,prevs,s,e);
                            lambsum += get_feature_sum_j(i,j,s,e);
                            double newvalue=logsumexp(table_A.get(i+di,j+dj,
                                        prevs)+lambsum+table_B.get(i,j,s),
                                    table_C.get(i,j,prevs,s));
                            table_C.set(newvalue,i,j,prevs,s);
                        }
                    }
                }
            }
        }
    }
}

//multiplies the forward and backward tables A and B to find marginals 
void Crf_Ed::fill_table_D(){
    for (int i=0;i<words->I;i++){
        for(int j=0;j<words->J;j++){
            for (int s=0;s<S;s++){
                double total=(table_A.get(i,j,s)+table_B.get(i,j,s));
                table_D.set(total,i,j,s);
            }
        }
    }
}

//public:

//get features active for i,j posisions in words
vector<int>* Data_Element::get_features(int i,int j){ 
   return &features[i][j];
}

//return first string of input string pair
string Data_Element::get_w1(){
    return w1;
}

//return second string of input string pair
string Data_Element::get_w2(){
    return w2;
}

//helper to return TRUE for vowel characters
bool is_vowel(char w){
    if (w=='a' || w=='e' || w=='i'||w=='o'||w=='u'){
        return true;
    }
    return false;
}

//for an input string pair (w1,w2) extract list of features and return as
//Data_Element
//(TODO: move to different executable - leave crf_ed program agnostic of
// feature extraction process)
Data_Element::Data_Element(string w1,string w2,set<int>& allowed_features)
    :w1(w1),w2(w2){    

    I=(int)w1.length();
    J=(int)w2.length();
    for (int i=0;i<I;i++){
        vector<vector<int> > jvec;
        for (int j=0;j<J;j++){
            vector<int> fvec;
            fvec.push_back(0);  //constant term
            if ((i>=1||j>=1)){
                //fvec.push_back(1);  //not beginning
            }
            if (i!=I-1 || j!=J-1){
                fvec.push_back(2);  //not ending
            }
            if (i==I-1 && j==J-1){
                fvec.push_back(3);  //ending
            }
            if (w2[j]=='#'){
                //fvec.push_back(4);  //second is #
            }
            if (w1[i]=='#' && w2[j]=='#'){
                fvec.push_back(5);  //both is #
            }
            if (w1[i]==w2[j]){
                fvec.push_back(6);  //character match
            }
            if (w1[i]!=w2[j]){
                fvec.push_back(7);  //character mismatch
            }
            if (j<J-1&&i<I-1&&w1[i+1]==w2[j] && w1[i]==w2[j+1]){
                fvec.push_back(8);  //character swop
            }
            if (j>0&&i>0&&w1[i]==w2[j-1] && w1[i-1]==w2[j]){
                fvec.push_back(9);  //character swop
            }
            if (is_vowel(w1[i])&&is_vowel(w2[j])){
                fvec.push_back(10);  //current are vowels
            }
            if (!(is_vowel(w1[i])&&is_vowel(w2[j]))){
                fvec.push_back(11);  //not both are vowels
            }
            if (!is_vowel(w1[i])&&!is_vowel(w2[j])){
                fvec.push_back(12);  //both not vowel
            }
            if (is_vowel(w1[i])&&!is_vowel(w2[j])){
                fvec.push_back(13); //vowel to consonant
            }
            if (!is_vowel(w1[i])&&is_vowel(w2[j])){
                fvec.push_back(14); //consonant to vowel
            }
            if (i>0&&w1[i-1]==w2[j]){
                fvec.push_back(15); //prev char in w1 same as this letter in w2
            }
            if (j>0&&w1[i]==w2[j-1]){
                fvec.push_back(16); //prev char in w1 same as this letter in w2
            }
            if (i>0&& w1[i]==w1[i-1]){
                fvec.push_back(17); //prev char in first = current in first
            }
            if (j>0&& w2[j]==w2[j-1]){
                fvec.push_back(18); //prev letter in second = current in second
            }
            for(vector<int>::iterator fi=fvec.begin();fi<fvec.end();fi++){
                if (allowed_features.find(*fi)==allowed_features.end()){
                    fvec.erase(fi);
                }
            }
            jvec.push_back(fvec);
        }
        features.push_back(jvec);
    }
}

//TODO: implement alternative Data_Element initializations
Data_Element::Data_Element(vector<vector<vector<int> > >){}
Data_Element::Data_Element(int I,int J,string list_of_features){}

//public: 

//new potential table, set all elements to initial_value
PTable::PTable(int initial_value,int max_I,int max_J,int max_S0,int max_S1)
    :initial_val(initial_value){
    I=max_I;
    J=max_J;
    S0=max_S0;
    S1=max_S1;
    for(int i=0;i<max_I;i++){
       vector<vector<vector<double> > > jvector;
       for(int j=0;j<max_J;j++){
           vector<vector<double> > s0vector;
           for(int s0=0;s0<max_S0;s0++){
               vector<double> s1vector;
               for(int s1=0;s1<max_S1;s1++){
                   s1vector.push_back(initial_value);
               }
               s0vector.push_back(s1vector);
           }
           jvector.push_back(s0vector);
       }
       table.push_back(jvector);
   }
}

//Reset potential table elements to initial_val. 
//Should be called before inference.
void PTable::reset(){
    for (int i=0;i<I;i++){
        for(int j=0;j<J;j++){
            for(int s0=0;s0<S0;s0++){
                for(int s1=0;s1<S1;s1++){
                    table[i][j][s0][s1]=initial_val;
                }
            }
        }
    }
}

//accessors for potential table
double PTable::get(int i,int j,int s0, int s1){
    return table[i][j][s0][s1];
}
double PTable::get(int i,int j,int s){ //assumes s1==0
    return table[i][j][s][0];
}
double PTable::get(int i,int j){       //assumes s0==s1==0
    return table[i][j][0][0];
}

//mutators for potential table
void PTable::set(double value,int i,int j,int s0, int s1){
    table[i][j][s0][s1]=value;
}
void PTable::set(double value,int i,int j,int s){         //assumes s1==0
    table[i][j][s][0]=value;
}
void PTable::set(double value,int i,int j){                //assumes s0==s1==0
    table[i][j][0][0]=value;
}

//for debugging
void PTable::print_table(){
    print_table(true);
}

//either prints in log or non-log domain
void PTable::print_table(bool exponen){
    for (int s1=0;s1<S1;s1++){
        cerr<<"S1 "<<s1<<endl;
        for(int s0=0;s0<S0;s0++){
            cerr<<"S0 "<<s0<<endl;
            for(int i=0;i<I;i++){
                for(int j=0;j<J;j++){
                    if (exponen){
                        cerr<<exp(table[i][j][s0][s1])<<" ";
                    }else{
                        cerr<<(table[i][j][s0][s1])<<" ";
                    }
                }
                cerr<<endl;
            }
        }
    }
}
