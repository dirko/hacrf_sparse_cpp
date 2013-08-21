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
double _lambdaparameter=1;
int _maxdataelements=-1;
int _numepochs;
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

double LNINF=log(0);

//helper
int convertString2Int(string number){
    int numb;
    istringstream(number)>>numb;
    return numb;
}

//helper
double convertString2Double(string number){
    double numb;
    if (number=="-inf")
        return LNINF;
    istringstream(number)>>numb;
    return numb;
}

//read datapoints
void readXFromFile(vector<vector<string> > &res, string filename){
    ifstream* in;
    try{
        in=(new ifstream(filename.c_str()));
    }
    catch (...){
        cerr<<"error reading file : "<<filename<<endl;
    }
    string line;
    while(getline(*in,line)){
        vector<string> x;
        istringstream liness(line);
        string tok;
        while(getline(liness,tok,'|')){
            x.push_back(tok);
        }
        res.push_back(x);
    }
    in->close();
}

//get parameters from file
void load_parameters(Crf_Ed& edo,string filename){
    ifstream* in;
    try{
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
        i++;
    }
    in->close();
}

//reads the set of features to be used
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
        allowed.insert(convertString2Int(line));
    }
    in->close();
    return allowed;
}

//write params to file
void save_parameters(Crf_Ed& edo,string filename){
    ofstream file(filename.c_str(),ios::out);
    vector<double>* params = edo.get_params();
    int P=(int)params->size();
    for (int i=0;i<P;i++){
        file<<(*params)[i]<<endl;
    }
    file.close();
}

//write options to file
void log_options(){
    _log<<"inputfileM"<< _inputfileM<<endl;
    _log<<"inputfileN"<< _inputfileN<<endl;
    _log<<"valfileM"<< _valfileM<<endl;
    _log<<"valfileN"<< _valfileN<<endl;
    _log<<"featurefile"<< _featurefile<<endl;
    _log<<"logfile"<< _logfile<<endl;
    _log<<"parameterfile"<< _parameterfile<<endl;
    _log<<"lambdaparameter"<< _lambdaparameter<<endl;
    _log<<"maxdataelements"<< _maxdataelements<<endl;
    _log<<"numepochs"<< _numepochs<<endl;
}

//prints usage
void usage(){
    cerr << "usage: learning [options]" << endl;
    cerr << "options:" << endl;
    cerr << "-S     : Score the string pairs in infileA and infileB" << endl;
    cerr << "-iM infileA   : Sets the input file for class A" << endl;
    cerr << "-iN infileB   : Sets the input file for class B" << endl;
    cerr << "-iV valfileA  : Sets the validation file for class A" << endl;
    cerr << "-iW valfileB  : Sets the validation file for class B" << endl;
    cerr << "-iF featfile  : Sets the file with a newline separated" << endl;
    cerr << "                list of features to include" << endl;
    cerr << "-oL logfile   : Sets the file where logs are written" << endl;
    cerr << "-oP paramfile : Sets the file where parameters are stored" << endl;
    cerr << "-lL regparam  : Sets the regularization parameter to " << endl;
    cerr << "                regparam. Default is 0 (no regularization)"<< endl;
    cerr << "-lE ep        : Trains model for maximum of ep epochs" << endl;
    cerr << "-lG           : Train model (using lm-bfgs)" << endl;
    cerr << "-mM maxl      : Only include example strings of length up" << endl;
    cerr << "                maxl. Default is -1 (include all)" << endl;
    exit(1);
}

//read and set command line options
void setOptions(int argc,char* argv[]){
    if (argc <= 1){
        usage();
    }
    _uselbfgs = false;
    for(int i=1;i<argc;i++){
        switch(argv[i][1]){
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
                    case 'L':
                        _lambdaparameter=atof(argv[i+1]);
                        i++;
                        break;
                    case 'E':
                        _numepochs=atoi(argv[i+1]);
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
                    
//the function that is passed to optimization routine. calculate opjective
//function (negative log-likelihood) and its derivative
void function_grad(const real_1d_array &x,
        double &func,real_1d_array &grad,void *ptr){
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
    }
    for (int cx=z1start;cx<z1end;cx++){
        edo.set_words(&dat_z1[cx]);
        edo.fill_tables();
        correct+=edo.get_accuracy(1);
        total++;
        edo.get_derivs(1,dlamb);
        ll+=edo.get_ll(1);
    }
    j=0;
    double penalty=0;
    for (int i=0;i<(int)lamb->size();i++){
        if (!isinf(-lamb->at(i))){
            grad[j]=-(dlamb[i]-lambda*(*lamb)[i]);
            penalty+= lambda*(*lamb)[i]*(*lamb)[i]*0.5;
            if (isnan(dlamb[i])||isinf(-dlamb[i])){
                grad[j]=0;
            }
            if (i<=26){ //sets topology params to unchangegable
                grad[j]=0;
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

//evaluates the current model on datapoints dat_z0 and dat_z1
double evaluate(Crf_Ed&edo,
        vector<Data_Element>& dat_z0,vector<Data_Element>& dat_z1){
    double correct=0;
    double total=0;
    vector<double>conf(4,0);
    for (int x=0;x<(int)dat_z0.size();x++){
        edo.set_words(&dat_z0[x]);
        edo.fill_table_ll();
        correct+=edo.get_accuracy(0);
        double z0,z1;
        edo.evaluate(0,z0,z1);
        total+=1;
        conf[edo.get_accuracy(0)]+=1;
    }

    for (int x=0;x<(int)dat_z1.size();x++){
        edo.set_words(&dat_z1[x]);
        edo.fill_table_ll();
        correct+=edo.get_accuracy(1);
        double z0,z1;
        edo.evaluate(1,z0,z1);
        total+=1;
        conf[2+edo.get_accuracy(0)]+=1;
    }
    cerr<<"accuracy "<<correct<<" "<<total<<"   "<<correct/total<<endl;
    cerr<<conf[0]<<" "<<conf[1]<<endl;
    cerr<<conf[2]<<" "<<conf[3]<<endl;
    return correct/total;
}

//report function that is passed to optimization routine. logs current model
//accuracy, validation accuracy, and objective function every iteration
void function_report(const real_1d_array &x,double func,void *ptr){
    double acc = evaluate(_edo,_dat_zv0,_dat_zv1);
    cerr<<"["<<_epoch<<","<<_iteration<<","<<acc<<","<<_trainacc<<","<<
        func<<"],"<<endl;
    _log<<"["<<_epoch<<","<<_iteration<<","<<acc<<","<<_trainacc<<","<<
        func<<"],"<<endl;
    _iteration++;
}


int main(int argc,char* argv[]){
    setOptions(argc,argv);
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
    
    cerr<<"reading allowed features : "<<_featurefile<<endl;
    set<int> allowed_features = get_allowed_features(_featurefile);
    cerr<<"adding data elements: matches"<<endl;
    int maxdataelements=_maxdataelements;
    if (_maxdataelements<0){
        maxdataelements=fmax((int)raw_data_X0.size(),(int)raw_data_X1.size());
        maxdataelements=fmax(maxdataelements,(int)raw_data_Xv0.size());
        maxdataelements=fmax(maxdataelements,(int)raw_data_Xv1.size());
    }
    cerr<<"maxelements: "<<maxdataelements<<endl;
    for (int i=0;i<maxdataelements && i<(int)raw_data_X0.size();i++){
        Data_Element w(raw_data_X0[i][0],raw_data_X0[i][1],allowed_features);
        data_X0.push_back(w);
    }
    cerr<<"adding data elements: mismatches"<<endl;
    for (int i=0;i<maxdataelements && i<(int)raw_data_X1.size();i++){
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

    Crf_Ed edo(41,41,9,19);
    load_parameters(edo,_parameterfile);

    double valaccuracy=evaluate(edo,data_Xv0,data_Xv1);
    cerr<<"Validation accuracy: "<<endl;
    cout<<valaccuracy<<endl;

    _log<<"[epoch,iteration,val_accuracy,test_accuracy,LL]"<<endl;
    _log<<"["<<endl;
    if (_uselbfgs){
        cerr<<"starting training"<<endl;
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
        int nas=0;
        for (int i=0;i<(int)edo.get_params()->size();i++){
            if (!isinf(-(*edo.get_params())[i])){
                x0[nas] = (*edo.get_params())[i];
                nas++;
            }
        }
        minlbfgsstate state;
        minlbfgsreport rep;
        minlbfgscreate(4,x0,state);
        minlbfgssetcond(state,0.01,0.01,0.01,_numepochs);
        minlbfgssetxrep(state, true);
        alglib::minlbfgsoptimize(state,function_grad,function_report);
        minlbfgsresults(state,x0,rep);
        save_parameters(_edo,_parameterfile);
        edo = _edo;
        
    }
    if(_score){
        for (int x=0;x<(int)data_X0.size();x++){
            edo.set_words(&data_X0[x]);
            edo.fill_table_ll();
            double z0,z1;
            edo.evaluate(0,z0,z1);
            double z=z0/(z0+z1);
            cout<<z<<endl;
        }
        for (int x=0;x<(int)data_X0.size();x++){
            edo.set_words(&data_X0[x]);
            edo.fill_table_ll();
            double z0,z1;
            edo.evaluate(1,z0,z1);
            double z=z1/(z0+z1);
            cout<<z<<endl;
        }
    }
    _log<<"]"<<endl;
    _log.close();

    return 0;
}
