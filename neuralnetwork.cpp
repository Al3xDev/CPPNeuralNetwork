/*
Copyright 2020 Stoica Alexandru-Gabriel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

*/


#include "neuralnetwork.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

#define LOG_AND_EXIT(msg) logError(__FILE__,__LINE__, msg, true);
#define LOG(msg) logError(__FILE__,__LINE__,msg, false);
#define DEBUG(msg) { std::cerr<<"Debug: line "<<__LINE__<<"   " <<msg<<"\n";}
void logError( char* file_name, int line, char* msg, bool exit_b)
{
    std::cerr<<"["<<file_name<<"] - "<<line<<" - "<<msg<<"\n";
    if(exit_b)
    {
        exit(1);
    }
}

///Return a random number between -1 and 1
float nn_random() // it use RDTSC
{
    uint64_t ret;
    asm volatile ( "rdtsc" : "=A"(ret) );
    return 1.0 - 2*(ret%1000000)/1000000.0;
}

float lin_func( float val)
{
    return val;
}

float deriv_lin_func( float val)
{
    return 1;
}

float tanh_func( float val)
{
    return tanh(val);
}

float deriv_tanh_func( float val)
{
    return 1-val*val;
}

nn_connection::nn_connection()
{
    dest = 0;
    src = 0;
    weight = 0;
    delta_weight = 0;
}

nn_neuron::nn_neuron() {}

void nn_neuron::calc_sum_in()
{
    in = 0;

    for( int i=0; i<in_conn.size(); ++i)
    {
        in = in + (in_conn[i]->weight * in_conn[i]->src->out);
    }

    in = in + bias;
}

void nn_neuron::calc_out()
{
    out = f(in);
}

void nn_neuron::calc_deriv()
{
    deriv = deriv_f(in);
}

nn_perceptron::nn_perceptron()
{
    error = 0;
    status = NN_PERCEPTRON_ALIVE;
}

nn_perceptron::nn_perceptron( std::vector<int> &_topology)
{
    create( topology);
}


void nn_perceptron::create( std::vector<int> &_topology)
{
    learning_speed = 0.3;
    permitted_error = 0.1;

    if( _topology.size() < 3)
    {
        LOG_AND_EXIT( "Topology size < 3");
    }

    if( status == NN_PERCEPTRON_ALIVE)
    {
        destroy();
    }

    // copy the topology
    topology.resize(_topology.size());
    for( int i=0; i<_topology.size(); ++i)
    {
        topology[i] = _topology[i];
    }


    int layers_num = topology.size();
    int n = 0; // neurons count
    int conn = 0; // connection count bewteen 2 layers

    layers.resize( topology.size());

    for( int i=0; i<layers_num; ++i)
    {
        layers[i].n_ptr.resize(topology[i]);

        n = n + topology[i];
        if( i>0)
        {
            conn = conn + topology[i-1]*topology[i];
        }
    }

    // alloc memeory
    neurons.resize(n);
    connection.resize(conn);
    inputs_nptr.resize( topology[0]);
    outputs_nptr.resize( topology[ layers_num-1]);

    int cl_offset = 0;
    int c_index = 0;

    // init connections
    for( int i=0; i<layers_num-1; ++i)
    {
        for( int j=0; j<topology[i]; ++j)
        {
            for( int k=0; k<topology[i+1]; ++k)
            {
                connection[ c_index].src = &neurons[ cl_offset + j];
                connection[ c_index].dest = &neurons[ cl_offset + topology[i] + k];
                connection[ c_index].weight = nn_random();
                connection[ c_index].delta_weight = 0;

                ++c_index;
            }
        }
        cl_offset = cl_offset + topology[i];
    }


    // init neurons
    int ID=0;
    // init first layer
    for( int i=0; i<topology[0]; ++i)
    {
        inputs_nptr[i] = &neurons[ID];
        layers[0].n_ptr[i] = &neurons[ID];
        neurons[ID].ID = ID;
        neurons[ID].f = lin_func;
        neurons[ID].deriv_f = deriv_lin_func;
        neurons[ID].bias = 0.01;
        ++ID;
    }

    // init the hidden neurons
    // using the connections
    uint64_t neurons_size = neurons.size();
    for( int i=1; i<layers_num-1; ++i)
    {
        for( int j=0; j<topology[i]; ++j)
        {
            layers[i].n_ptr[j] = &neurons[ID];
            neurons[ID].ID = ID;
            neurons[ID].f = tanh_func;
            neurons[ID].deriv_f = deriv_tanh_func;
            neurons[ID].bias = 0.01;
            ++ID;
        }
    }

    // init the last layer
    for( int i=0; i<topology[ layers_num-1]; ++i)
    {
        outputs_nptr[i] = &neurons[ID];
        layers[ layers_num-1].n_ptr[i] = &neurons[ID];
        neurons[ID].ID = ID;
        neurons[ID].f = lin_func;
        neurons[ID].deriv_f = deriv_lin_func;
        neurons[ID].bias = 0.01;
        ++ID;
    }

    // link the neurons
    for( int i=0; i<connection.size(); ++i)
    {
        connection[i].src->out_conn.push_back( &connection[i]);
        connection[i].dest->in_conn.push_back( &connection[i]);
    }

    status = NN_PERCEPTRON_ALIVE;
}

void nn_perceptron::load_from_file( char* file_name)
{

}

void nn_perceptron::forward( std::vector<float> &in, std::vector<float> &out)
{
    if( status != NN_PERCEPTRON_ALIVE)
    {
        LOG( "The perceptron is not init");
    }

    for( int i=0; i<inputs_nptr.size(); ++i)
    {
        inputs_nptr[i]->in = in[i]; // input set
        inputs_nptr[i]->calc_out(); // calc first layer;
    }

    for( int i=1; i<layers.size(); ++i)
    {
        for( int j=0; j<layers[i].n_ptr.size(); ++j)
        {
            layers[i].n_ptr[j]->calc_sum_in();
            layers[i].n_ptr[j]->calc_out();
        }
    }

    for( int i=0; i<outputs_nptr.size(); ++i)
    {
        out[i] = outputs_nptr[i]->out;
    }
}

void nn_perceptron::train( std::vector<float> &in, std::vector<float> &out_exp)
{
    if( in.size() != inputs_nptr.size())
    {
        LOG( "Input data != input layer size\n  Failed training");
        return;
    }

    if( out_exp.size() != outputs_nptr.size())
    {
        LOG( "Output data != output layer size\n  Failed training");
        return;
    }


    //Back Propagation


    // out_exp is expected output
    std::vector<float> out(out_exp.size()); //predict. out
    // predict
    forward( in, out);
    // get the error and write it in last neurons layer
    error = 0;
    float temp;
    for( int i=0; i<out.size(); ++i)
    {
        temp = out_exp[i] - out[i];
        outputs_nptr[i]->error = temp;
        error = error + temp*temp;

        // get deriv. for output layer
        outputs_nptr[i]->calc_deriv();
    }

    if( error > permitted_error)
    {
        nn_neuron *n, *m;
        nn_connection *c;
        for( int i=layers.size()-2; i>=0; --i)
        {
            for( int j=0; j<layers[i].n_ptr.size(); ++j)
            {
                n = layers[i].n_ptr[j];
                n->error = 0;
                // calculate the error
                for( int k=0; k<n->out_conn.size(); ++k)
                {
                    c = n->out_conn[k];

                    // prepare for next layer
                    n->calc_deriv();

                    // get the error
                    m = n->out_conn[k]->dest;
                    n->error = n->error + m->error*m->deriv *n->out_conn[k]->weight;

                    // calculate the delta_weight
                    c->delta_weight = c->delta_weight - learning_speed*m->error*m->deriv*c->src->out;

                    c->update(); // update the weight
                }
            }
        }
    }
}

void nn_perceptron::print()
{

    nn_neuron *n;
    std::cerr<<"nn_perceptron print - use layers\n";
    for( int i=0; i<layers.size(); ++i)
    {
        for( int j=0; j<layers[i].n_ptr.size(); ++j)
        {
            std::cerr<<"Neuron\n";
            n = layers[i].n_ptr[j];
            std::cerr<<" ID: "<<n->ID<<"\n";
            std::cerr<<" in: "<<n->in<<"\n";
            std::cerr<<" out: "<<n->out<<"\n";
        }
    }

}


void nn_perceptron::destroy()
{
    status = NN_PERCEPTRON_DEAD;
    topology.clear();
    layers.clear();
    neurons.clear();
    connection.clear();
    inputs_nptr.clear();
    outputs_nptr.clear();
}

nn_perceptron::~nn_perceptron()
{
    destroy();
}
