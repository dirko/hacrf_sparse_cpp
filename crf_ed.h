#include <vector>
#include <iostream>
#include <string>
#include <math.h>
#include <iostream>
#include <set>
//#define endl "\n"
class Data_Element;
class Transition;

using std::vector;
using std::set;
using std::string;
using std::pair;

class PTable{
    /* PTable to store values for forward-backward algorithm
     */
    public: 
        PTable(int initial_value,int max_I,int max_J,int max_S0,int max_S1);
        void reset();
        double get(int i,int j,int s0, int s1);
        double get(int i,int j,int s);         //assumes s1==0
        double get(int i,int j);                //assumes s0==s1==0
        void set(double value,int i,int j,int s0, int s1);
        void set(double value,int i,int j,int s);         //assumes s1==0
        void set(double value,int i,int j);                //assumes s0==s1==0
        void print_table();      //prints table to cerr for debugging
        void print_table(bool);      //prints table to cerr for debugging
        int get_max_I(){return I;}
        int get_max_J(){return J;}
    private:
        vector<vector<vector<vector<double> > > > table;   //accessed: table[i][j][s0][s1] 
        double initial_val;
        int I;
        int J;
        int S0;
        int S1;
};

class Crf_Ed{
    /* Conditional Random Field Edit distance inference engine
     */
    public:
        void evaluate(int y,double & z0, double & z1);  //fills forward table to evaluate params on current example
        void fill_tables();
        void set_words(Data_Element*);
        void fill_table_ll();
        double get_ll(int);
        double get_accuracy(int);
        void get_derivs(int,vector<double> & derivs);    //adds derivatives to vector
        void init_params();
        vector<double>* get_params();
        Crf_Ed(int,int,int,int);

    private:
        void fill_table_A();
        void fill_table_B();
        void fill_table_C();
        void fill_table_D();
        Data_Element* words;    //current words
        int S;          //number of states
        int O;          //number of features
        int E;          //number of transitions
        vector<pair<int,int> > transitions;   //state machine
        PTable table_A;  //forward table
        PTable table_B;  //backward table
        PTable table_C;  //transision table
        PTable table_D;  //marginal table
        int end_state_match;    //the end state for the match transducer
        int end_state_mismatch;    //the end state for the mismatch transducer
        vector<double> params;  //parameters
        double logsumexp(double x,double y);
        void get_features_j(int i,int j,int s,int e);
        vector<int> vj;         //stores current features
        int vj_length;          //number of active features
        double get_feature_sum_j(int i,int j,int s,int e);
        void get_features_ij(int i,int j,int ps,int s,int e);
        vector<int> vij;        //stores current features
        int vij_length;          //number of active features
        double get_feature_sum_ij(int i,int j,int ps,int s,int e);
        bool right_transducer(int z,int state); //helper function to see if state belong to match or mismach transducer
};

class Data_Element{
    /* A single word pair
     */
    public:
        int I;  //length of first word
        int J;  //length of second word
        vector<int>* get_features(int i,int j); //gets features active for i,j posisions in words
        Data_Element(string w1,string w2,set<int>&);
        Data_Element(vector<vector<vector<int> > >);
        Data_Element(int I,int J,string list_of_features);
        string get_w1();
        string get_w2();
    private:
        vector<vector<vector<int> > > features; //features. accessed with features[i][j] 
        string w1;
        string w2;
};

