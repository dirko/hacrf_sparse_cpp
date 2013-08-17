#include "crf_ed.h"
#include <stdlib.h>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <set>
#include <unistd.h>
#include "alglib/src/stdafx.h"
#include "alglib/src/optimization.h"

//#include "dlib/stdafx.h"
//#include "dlib/optimization.h"

using namespace std;
using alglib::real_1d_array;
using alglib::minlbfgsstate;
using alglib::minlbfgscreate;
using alglib::minlbfgsreport;
using alglib::minlbfgssetcond;
using alglib::minlbfgsoptimize;
using alglib::minlbfgsresults;

string _inputfileM;
string _inputfileN;
string _valfileM;
string _valfileN;
string _featurefile;
string _logfile;
ofstream _log;
string _parameterfile;
double _learnrate;
double _lambdaparameter=1;
int _maxdataelements=-1;
int _numepochs;
int _batchsize;
bool _uselbfgs;
Crf_Ed _edo(40,40,9,19);
vector<Data_Element> _dat_z0;
vector<Data_Element> _dat_z1;
vector<Data_Element> _dat_zv0;
vector<Data_Element> _dat_zv1;
int _epoch=0;
int _iteration=0;
double _ll=0;
double _trainacc=0;
bool _score=false;
bool _obj_grad=false;
bool _obj_eval=false;

double LNINF=log(0);

int convertString2Int(string number){
    int numb;
    istringstream(number)>>numb;
    return numb;
}
double convertString2Double(string number){
    double numb;
    if (number=="-inf")
        return LNINF;
    istringstream(number)>>numb;
    return numb;
}
void readXFromFile(vector<vector<string> > &res, string filename){
    ifstream* in;
    try{
        //cerr<<"reading file "<<filename<<endl;
        in=(new ifstream(filename.c_str()));
    }
    catch (...){
        cerr<<"error reading file : "<<filename<<endl;
    }
    string line;
    while(getline(*in,line)){
        //cerr<<line<<endl;
        vector<string> x;//=new vector<string>();
        istringstream liness(line);
        string tok;
        while(getline(liness,tok,'|')){
            //string * newstring=new string(tok);
            x.push_back(tok);
        }
        res.push_back(x);
    }
    in->close();
}
void load_parameters(Crf_Ed& edo,string filename){
    ifstream* in;
    try{
        //cerr<<"reading parameters from file : "<<filename<<endl;
        in=(new ifstream(filename.c_str()));
    }
    catch(...){
        cerr<<"error reading file"<<endl;
    }
    string line;
    int i=0;
    vector<double>* params=edo.get_params();
    while (getline(*in,line)){
        (*params)[i]=convertString2Double(line);
        //cerr<<"line "<<i<<"  "<<line<< "   "<<convertString2Double(line)<<endl;;
        i++;
    }
    in->close();
}
set<int> get_allowed_features(string filename){
    ifstream* in;
    set<int> allowed;
    try{
        in=(new ifstream(filename.c_str()));
    }
    catch(...){
        cerr<<"error reading file"<<endl;
    }
    string line;
    while (getline(*in,line)){
        //(*params)[i]=convertString2Double(line);
        allowed.insert(convertString2Int(line));
        //cerr<<"line "<<i<<"  "<<line<< "   "<<convertString2Double(line)<<endl;;
    }
    in->close();
    return allowed;
}

void save_parameters(Crf_Ed& edo,string filename){
    ofstream file(filename.c_str(),ios::out);
    //cerr<<"writing params to file : "<<filename<<" ... ";
    vector<double>* params = edo.get_params();
    int P=(int)params->size();
    for (int i=0;i<P;i++){
        file<<(*params)[i]<<endl;
    }
    file.close();
    //cerr<<" written."<<endl;
}

void log_options(){
    _log<<"inputfileM"<< _inputfileM<<endl;
    _log<<"inputfileN"<< _inputfileN<<endl;
    _log<<"valfileM"<< _valfileM<<endl;
    _log<<"valfileN"<< _valfileN<<endl;
    _log<<"featurefile"<< _featurefile<<endl;
    _log<<"logfile"<< _logfile<<endl;
    _log<<"parameterfile"<< _parameterfile<<endl;
    _log<<"learnrate"<< _learnrate<<endl;
    _log<<"lambdaparameter"<< _lambdaparameter<<endl;
    _log<<"maxdataelements"<< _maxdataelements<<endl;
    _log<<"numepochs"<< _numepochs<<endl;
    _log<<"batchsize"<< _batchsize<<endl;
}


void setOptions(int argc,char* argv[]){
    _uselbfgs = false;
    for(int i=1;i<argc;i++){
        switch(argv[i][1]){
            case 'g':
                switch (argv[i][2]){
                    case 'E':
                        _obj_eval=true;
                        break;
                    case 'B':
                        _obj_grad=true;
                        break;
                }
                break;
            case 'S':
                _score=true;
                break;
            case 'i':
                switch (argv[i][2]){
                    case 'M':
                        _inputfileM=argv[i+1];
                        i++;
                        break;
                    case 'N':
                        _inputfileN=argv[i+1];
                        i++;
                        break;
                    case 'V':
                        _valfileM=argv[i+1];
                        i++;
                        break;
                    case 'W':
                        _valfileN=argv[i+1];
                        i++;
                        break;
                    case 'F':
                        _featurefile=argv[i+1];
                        i++;
                        break;
                }
                break;
            case 'o':
                switch(argv[i][2]){
                    case 'L':
                        _logfile=argv[i+1];
                        _log.open(_logfile.c_str());
                        i++;
                        break;
                    case 'P':
                        _parameterfile=argv[i+1];
                        i++;
                        break;
                }
            case 'l':
                switch (argv[i][2]){
                    case 'R':
                        _learnrate=atof(argv[i+1]);
                        i++;
                        break;
                    case 'L':
                        _lambdaparameter=atof(argv[i+1]);
                        i++;
                        break;
                    case 'E':
                        _numepochs=atoi(argv[i+1]);
                        i++;
                        break;
                    case 'B':
                        _batchsize=atoi(argv[i+1]);
                        i++;
                        break;
                    case 'G':
                        _uselbfgs=true;
                        break;
                }
            case 'm':
                switch (argv[i][2]){
                    case 'M':
                        _maxdataelements=atoi(argv[i+1]);
                        i++;
                        break;
                }
        }
    }
}
                    
double learning_update(Crf_Ed& edo,vector<Data_Element>& dat_z0,int z0start,int z0end,vector<Data_Element>& dat_z1,int z1start,int z1end,double learnrate,double lambda){
    double ll=0;
    vector<double>* lamb = edo.get_params();
    vector<double> dlamb((int)lamb->size());
    //for(int i=0;i<(int)lamb->size();i++){   //init dlamb with right length
    //    dlamb.push_back(0.0);
    //}
    
    for (int x=z0start;x<z0end;x++){
        edo.set_words(&dat_z0[x]);
        edo.fill_tables();
        edo.get_derivs(0,dlamb);
        ll+=edo.get_ll(0);
    }
    for (int x=z1start;x<z1end;x++){
        edo.set_words(&dat_z1[x]);
        edo.fill_tables();
        edo.get_derivs(1,dlamb);
        ll+=edo.get_ll(1);
    }
    for (int i=0;i<(int)lamb->size();i++){
        if (!isinf(-lamb->at(i)) && !isnan(dlamb[i]) && !isinf(-dlamb[i])){
            //cerr<<"NaN b "<<i<<" "<<(*lamb)[i]<<" "<<dlamb[i]<<"   "<<lamb->at(i)<<endl;
            (*lamb)[i] = (*lamb)[i] + learnrate*(dlamb[i] - lambda*(*lamb)[i]);
            if (isnan((*lamb)[i])){
            //    cerr<<"NaN a "<<i<<" "<<(*lamb)[i]<<" "<<dlamb[i]<<"   "<<lamb->at(i)<<endl;
            }
            if (lamb->at(i)>1){
                (*lamb)[i]=1;
            }
            if (lamb->at(i)<-1){
                (*lamb)[i]=-1;
            }
        }
    }
    return ll;
}
void function_grad(const real_1d_array &x,double &func,real_1d_array &grad,void *ptr){
    Crf_Ed edo=_edo;
    vector<Data_Element> dat_z0=_dat_z0;
    vector<Data_Element> dat_z1=_dat_z1;
    vector<double> *lamb=edo.get_params();
    double lambda=_lambdaparameter;
    int I=lamb->size();
    int j=0;
    for (int i=0;i<I;i++){
        if ((*lamb)[i]!=LNINF){
            (*lamb)[i]=x[j];
            j++;
        }
    }
    vector<double> dlamb((int)lamb->size());
    int z0start=0;
    int z0end=dat_z0.size();
    int z1start=0;
    int z1end=dat_z1.size();
    double ll=0;
    double correct=0;
    double total=0;
    for (int cx=z0start;cx<z0end;cx++){
        edo.set_words(&dat_z0[cx]);
        edo.fill_tables();
        correct+=edo.get_accuracy(0);
        total++;
        edo.get_derivs(0,dlamb);
        ll+=edo.get_ll(0);
        //cout<<ll<<" "<<x<<endl;
    }
    for (int cx=z1start;cx<z1end;cx++){
        edo.set_words(&dat_z1[cx]);
        edo.fill_tables();
        //edo.table_A.print_table();
        //edo.table_B.print_table();
        //edo.table_D.print_table();
        correct+=edo.get_accuracy(1);
        total++;
        edo.get_derivs(1,dlamb);
        ll+=edo.get_ll(1);
    }
    cerr<<"correct training "<<total<<" "<<correct<<endl;
    j=0;
    double penalty=0;
    for (int i=0;i<(int)lamb->size();i++){
        if (!isinf(-lamb->at(i))){// && !isnan(dlamb[i]) && !isinf(-dlamb[i])){
            //cerr<<"NaN b "<<i<<" "<<(*lamb)[i]<<" "<<dlamb[i]<<"   "<<lamb->at(i)<<endl;
            //(*lamb)[i] = (*lamb)[i] + learnrate*(dlamb[i] - lambda*(*lamb)[i]);
            grad[j]=-(dlamb[i]-lambda*(*lamb)[i]);
            penalty+= lambda*(*lamb)[i]*(*lamb)[i]*0.5;
            if (isnan(dlamb[i])||isinf(-dlamb[i])){
                grad[j]=0;
            //    cerr<<"NaN a "<<i<<" "<<(*lamb)[i]<<" "<<dlamb[i]<<"   "<<lamb->at(i)<<endl;
            }
            if (i<=26){ //sets topology params to unchangegable
                grad[j]=0;
            }
            if (grad[j]>100){
                //grad[j]=100;
            }
            if (grad[j]<-100){
                //grad[j]=-100;
            }
            j++;
        }
    }
    _ll=ll;
    _trainacc=correct/total;
    _edo=edo;
    func=-(double)ll+(double)penalty;
    _epoch++;
}
double evaluate(Crf_Ed&edo,vector<Data_Element>& dat_z0,vector<Data_Element>& dat_z1);

void function_report(const real_1d_array &x,double func,void *ptr){
    double acc = evaluate(_edo,_dat_zv0,_dat_zv1);
    cerr<<"["<<_epoch<<","<<_iteration<<","<<acc<<","<<_trainacc<<","<<func<<"],"<<endl;
    _log<<"["<<_epoch<<","<<_iteration<<","<<acc<<","<<_trainacc<<","<<func<<"],"<<endl;
    _iteration++;
}
double evaluate(Crf_Ed&edo,vector<Data_Element>& dat_z0,vector<Data_Element>& dat_z1){
    double correct=0;
    double total=0;
    vector<double>conf(4,0);
    //cerr<<"datsize "<<dat_z0.size()<<endl;
    for (int x=0;x<(int)dat_z0.size();x++){
        edo.set_words(&dat_z0[x]);
        edo.fill_table_ll();
        correct+=edo.get_accuracy(0);
        double z0,z1;
        edo.evaluate(0,z0,z1);
        //if (isnan(-z0)){
        //    edo.table_A.print_table(false);
            //sleep(3);
        //}
        cerr<<"eval "<<dat_z0[x].get_w1()<<"|"<<dat_z0[x].get_w2()<<x<<" "<<z0<<" "<<z1<<"    "<<(z0>z1)<<"  lL "<<_lambdaparameter<< endl;
        total+=1;
        conf[edo.get_accuracy(0)]+=1;
    }

    for (int x=0;x<(int)dat_z1.size();x++){
        edo.set_words(&dat_z1[x]);
        edo.fill_table_ll();
        correct+=edo.get_accuracy(1);
        double z0,z1;
        edo.evaluate(1,z0,z1);
        cerr<<"eval "<<dat_z1[x].get_w1()<<"|"<<dat_z1[x].get_w2()<<x<<" "<<z0<<" "<<z1<<"    "<<(z1>z0)<<endl;
        total+=1;
        conf[2+edo.get_accuracy(0)]+=1;
    }
    cerr<<"accuracy "<<correct<<" "<<total<<"   "<<correct/total<<endl;
    cerr<<conf[0]<<" "<<conf[1]<<endl;
    cerr<<conf[2]<<" "<<conf[3]<<endl;
    return correct/total;
}
double evaluate_objective(Crf_Ed&edo,const real_1d_array &x,vector<Data_Element>& dat_z0,vector<Data_Element>& dat_z1){
    vector<double> *lamb=edo.get_params();
    //double lambda=_lambdaparameter;
    int I=lamb->size();
    int j=0;
    for (int i=0;i<I;i++){
        if ((*lamb)[i]!=LNINF){
            (*lamb)[i]=x[j];
            j++;
        }
    }
    double ll=0;
    for (int cx=0;cx<(int)dat_z0.size();cx++){
        edo.set_words(&dat_z0[cx]);
        edo.fill_table_ll();
        ll+=edo.get_ll(0);
    }

    for (int cx=0;cx<(int)dat_z1.size();cx++){
        edo.set_words(&dat_z1[cx]);
        edo.fill_table_ll();
        ll+=edo.get_ll(1);
    }
    return ll;
}
void score(){
    //cout<<"kaas"<<endl;
    //return;
    //string sin;
    //if (cin){
    //    cin>>sin;
    //}
    vector<vector<string> > res;
    string line;
    while(getline(cin,line)){
        //cerr<<line<<endl;
        vector<string> x;//=new vector<string>();
        istringstream liness(line);
        string tok;
        while(getline(liness,tok,'|')){
            //string * newstring=new string(tok);
            x.push_back(tok);
        }
        res.push_back(x);
    }

    vector<Data_Element> data_X;
    set<int> allowed_features = get_allowed_features(_featurefile);
 
    int maxdataelements=_maxdataelements;
    if (_maxdataelements<0){
        maxdataelements=(int)res.size();
    }
    for (int i=0;i<maxdataelements && i<(int)res.size();i++){
        Data_Element w(res[i][0],res[i][1],allowed_features);
        data_X.push_back(w);
    }
    Crf_Ed edo(40,40,9,19);
    load_parameters(edo,_parameterfile);

    for (int x=0;x<(int)data_X.size();x++){
        edo.set_words(&data_X[x]);
        edo.fill_table_ll();
        double z0,z1;
        edo.evaluate(0,z0,z1);
        double z=z0/(z0+z1);
        //cout<<data_X[x].get_w1()<<" "<<data_X[x].get_w2()<<" "<<z<<endl;
        cout<<z<<endl;
    }
}

void pipe_grad(bool both){
    //Crf_Ed edo(6,8,9,19);
    Crf_Ed edo(30,30,9,19);
    //cerr<<"edo"<<endl;
    real_1d_array x;
    //cerr<<"edo1"<<endl;
    int as=0;
    for (int i=0;i<(int)edo.get_params()->size();i++){
        if (!isinf(-(*edo.get_params())[i])){
            as++;
        }
    }
    cerr<<"as "<<as<<endl;
    //cerr<<"edo2"<<endl;
    x.setlength(as);
    for (int i=0;i<as;i++){
        x[i]=0.0;
    }

    string line;
    int ci=0;
    while(getline(cin,line)){
        //cerr<<"kaads"<<endl;
        x[ci]=(convertString2Double(line));
        //cerr<<i<<x[i]<<endl;
        ci++;
    }
    vector<vector<string> > raw_data_X0;
    vector<Data_Element> data_X0;
    vector<vector<string> > raw_data_X1;
    vector<Data_Element> data_X1;
    readXFromFile(raw_data_X0,(_inputfileM));
    readXFromFile(raw_data_X1,(_inputfileN));
    set<int> allowed_features = get_allowed_features(_featurefile);
    int maxdataelements=_maxdataelements;
    if (_maxdataelements<0){
        maxdataelements=fmax((int)raw_data_X0.size(),(int)raw_data_X1.size());
    }
    for (int i=0;i<maxdataelements && i<(int)raw_data_X0.size();i++){
        Data_Element w(raw_data_X0[i][0],raw_data_X0[i][1],allowed_features);
        data_X0.push_back(w);
    }
    for (int i=0;i<maxdataelements && i<(int)raw_data_X1.size();i++){
        Data_Element w(raw_data_X1[i][0],raw_data_X1[i][1],allowed_features);
        data_X1.push_back(w);
    }
 
    //load_parameters(edo,_parameterfile);
    _edo=edo;
    _dat_z0 = data_X0;
    _dat_z1 = data_X1;
    double func=0.0;
    real_1d_array grad;
    grad.setlength(as);
    if (both){
        //cout<<data_X0.size()<<" "<<data_X1.size()<<" "<<maxdataelements<<" "<<_inputfileM<<" "<<_inputfileN<<endl;
        function_grad(x,func,grad,NULL);
        //double ll =evaluate_objective(edo,x,data_X0,data_X1);
        //func=-ll;

        cout<<func<<endl;
        cout<<_trainacc<<endl;
        //cout<<10<<endl;
        for (int t=0;t<as;t++){
            cout<<grad[t]<<endl;
        }
    }else{
        function_grad(x,func,grad,NULL);
        //double ll =evaluate_objective(edo,x,data_X0,data_X1);
        //func=-ll;

        cout<<func<<endl;
        //double ll =evaluate_objective(edo,x,data_X0,data_X1);
        //cout <<-ll<<endl;
    }
    cerr<<"saving "<<_parameterfile<<" "<<as<<" "<<data_X0.size()<<endl;
    save_parameters(_edo,_parameterfile);
}

int main(int argc,char* argv[]){
    setOptions(argc,argv);
    if (_score){    //no leaning is done, the input is scored
        score();
        return 0;
    }
    if (_obj_grad){
        pipe_grad(true);
        return 0;
    }
    if (_obj_eval){
        pipe_grad(false);
        return 0;
    }
    log_options();
    vector<vector<string> > raw_data_X0;
    vector<Data_Element> data_X0;
    vector<vector<string> > raw_data_X1;
    vector<Data_Element> data_X1;
    vector<vector<string> > raw_data_Xv0;
    vector<Data_Element> data_Xv0;
    vector<vector<string> > raw_data_Xv1;
    vector<Data_Element> data_Xv1;
    cerr<<"reading X "<<_inputfileM<<" "<<_valfileM<<endl;;
    readXFromFile(raw_data_X0,(_inputfileM));
    readXFromFile(raw_data_X1,(_inputfileN));
    readXFromFile(raw_data_Xv0,(_valfileM));
    readXFromFile(raw_data_Xv1,(_valfileN));
    //readXFromFile(raw_data_X0,string("testdata1.txt"));
    //readXFromFile(raw_data_X1,string("testdata2.txt"));
    cerr<<"reading allowed features : "<<_featurefile<<endl;
    set<int> allowed_features = get_allowed_features(_featurefile);
    cerr<<"adding data elements: matches"<<endl;
    int maxdataelements=_maxdataelements;
    if (_maxdataelements<0){
        //maxdataelements=(int)raw_data_Xv0.size();
        maxdataelements=fmax((int)raw_data_X0.size(),(int)raw_data_X1.size());
        maxdataelements=fmax(maxdataelements,(int)raw_data_Xv0.size());
        maxdataelements=fmax(maxdataelements,(int)raw_data_Xv1.size());
    }
    cerr<<"maxelements: "<<maxdataelements<<endl;
    for (int i=0;i<maxdataelements && i<(int)raw_data_X0.size();i++){
        //cerr<<"0 element "<<i<<"  "<<endl;
        //Data_Element *w = new Data_Element(raw_data_X[i][0],raw_data_X[i][1]);
        Data_Element w(raw_data_X0[i][0],raw_data_X0[i][1],allowed_features);
        data_X0.push_back(w);
    }
    cerr<<"adding data elements: mismatches"<<endl;
    for (int i=0;i<maxdataelements && i<(int)raw_data_X1.size();i++){
        //cerr<<"1 element "<<i<<"  "<<endl;
        //Data_Element *w = new Data_Element(raw_data_X[i][0],raw_data_X[i][1]);
        Data_Element w(raw_data_X1[i][0],raw_data_X1[i][1],allowed_features);
        data_X1.push_back(w);
    }
    cerr<<"adding data elements: validation matches"<<endl;
    for (int i=0;i<maxdataelements && i<(int)raw_data_Xv0.size();i++){
        Data_Element w(raw_data_Xv0[i][0],raw_data_Xv0[i][1],allowed_features);
        data_Xv0.push_back(w);
    }
    cerr<<"adding data elements: validation mismatches"<<endl;
    for (int i=0;i<maxdataelements && i<(int)raw_data_Xv1.size();i++){
        Data_Element w(raw_data_Xv1[i][0],raw_data_Xv1[i][1],allowed_features);
        data_Xv1.push_back(w);
    }

    Crf_Ed edo(30,30,9,19);
    load_parameters(edo,_parameterfile);

    double valaccuracy=evaluate(edo,data_Xv0,data_Xv1);
    cerr<<"Validation accuracy: "<<endl;
    cout<<valaccuracy<<endl;

    double learnrate=_learnrate;//0.001;
    double lambda=_lambdaparameter;//0.1;
    int numminibatches = data_X1.size()/_batchsize;
    _log<<"[epoch,iteration,val_accuracy,test_accuracy,LL]"<<endl;
    _log<<"["<<endl;
    if (!_uselbfgs){
    for (int epoch=0;epoch<_numepochs;epoch++){
        double LL=0;
        for(int minibatch=0;minibatch<numminibatches;minibatch++){
            int x0s=minibatch*_batchsize;
            int x0e=(minibatch+1)*_batchsize;
            int x1s=minibatch*_batchsize;
            int x1e=(minibatch+1)*_batchsize;
            if (x0e>(int)data_X0.size())
                x0e=data_X0.size();
            if (x1e>(int)data_X1.size())
                x1e=data_X1.size();
            double newlearnrate = learnrate/(x0e-x0s + x1e-x1s);
            LL += learning_update(edo,data_X0,x0s,x0e,data_X1,x1s,x1e,newlearnrate,lambda);
        }
        cerr<<"iteration "<<epoch<<" LL "<<LL<<endl;
        double valaccuracy=evaluate(edo,data_Xv0,data_Xv1);
        double testaccuracy=evaluate(edo,data_X0,data_X1);
        _log<<"["<<epoch<<","<<valaccuracy<<","<<testaccuracy<<","<<LL<<"],"<<endl;
        save_parameters(edo,_parameterfile);
    }
    }else{
        _edo=edo;
        _dat_z0 = data_X0;
        _dat_z1 = data_X1;
        _dat_zv0 = data_Xv0;
        _dat_zv1 = data_Xv1;
        real_1d_array x0;
        int as=0;
        for (int i=0;i<(int)edo.get_params()->size();i++){
            if (!isinf(-(*edo.get_params())[i])){
                as++;
            }
        }
        x0.setlength(as);
        for (int i=0;i<as;i++){
            x0[i]=0.0;
        }
        minlbfgsstate state;
        minlbfgsreport rep;
        minlbfgscreate(4,x0,state);
        minlbfgssetcond(state,0,0,0,_numepochs);
        minlbfgssetxrep(state, true);
        alglib::minlbfgsoptimize(state,function_grad,function_report);
        minlbfgsresults(state,x0,rep);
        save_parameters(_edo,_parameterfile);
        
    }
    _log<<"]"<<endl;
    _log.close();

    return 0;
}
