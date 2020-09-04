#include <iostream>
#include <vector>
#include "neuralnetwork.h"
using namespace std;

struct data_s
{
    float x1,x2;
    float y;
};


int main()
{
    vector<int> topology(3);
    topology[0] = 2;
    topology[1] = 1;
    topology[2] = 1;

    nn_perceptron *nnp = new nn_perceptron;
    nnp->create( topology);

    data_s *data = new data_s[4];

    data[0].x1=1;
    data[0].x2=1;
    data[0].y=1;


    data[1].x1=1;
    data[1].x2=0;
    data[1].y=1;


    data[2].x1=0;
    data[2].x2=1;
    data[2].y=1;

    data[3].x1=0;
    data[3].x2=0;
    data[3].y=0;



    vector<float> in(2);
    vector<float> out(1);

    for( int i=0; i<10000; ++i)
    {
        uint64_t ret;
        asm volatile ( "rdtsc" : "=A"(ret) );

        in[0] = data[ i%4].x1;
        in[1] = data[ i%4].x2;
        out[0] = data[ i%4].y;

        nnp->train( in, out);
        cerr<<"Error: "<<nnp->error<<"\n";
    }

    in[0] = 0.1;
    in[1] = 0.0;

    nnp->forward( in, out);


    cerr<<"\n^ "<<out[0]<<"\n^ "<<nnp->error<<"\n";


    return 0;
}
