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

#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#include <cstdint>
#include <vector>

class nn_neuron;
class nn_connection;
class nn_perceptron;

#define NN_CONNECTION_SRC 0
#define NN_CONNECTION_DEST 1


/** Use like a arrow with a sourse and a destination. <br>
 It keep the weight and the delta weight**/
class nn_connection
{
public:
	nn_connection();
	~nn_connection(){}

	/// Update the conections weight 
	void update() { weight = weight-delta_weight;}

	nn_neuron *src;
	nn_neuron *dest;

	float weight;
	float delta_weight;
};

class nn_neuron
{
public:
	nn_neuron();
	~nn_neuron(){ in_conn.clear(); out_conn.clear();}

	/// Weighted sum of previous layer out
	void calc_sum_in();
	/// out = f( in)
	void calc_out();
	/// deriv = deriv_f( in)
	void calc_deriv();

	/// Calculate the input sum and the output.
	void calc(){ calc_sum_in(); calc_out();}

	/** The activation function	<br>
	 If the neuron is in the first or the last layer the activation function will be linear function, <br>
	 else the activation function will be the hyperbolic tangent.**/
	float (*f)(float);
	/// The derivative of the activation function
	float (*deriv_f)(float);

    float bias; ///fixed
    float in;
	float out;
	float deriv;

	/// The error use for training
	float error;

	int ID;

	std::vector<nn_connection*> in_conn;
	std::vector<nn_connection*> out_conn;
};


class nn_layer
{
public:
	nn_layer(){}
	~nn_layer(){ n_ptr.clear();}
	// uint64_t type;
	/// Vector of pointer for the layer neurons
	std::vector<nn_neuron*> n_ptr;
};

#define NN_PERCEPTRON_ALIVE 1
#define NN_PERCEPTRON_DEAD 0


class nn_perceptron
{
public:
	//not use
	nn_perceptron();
	/// The constructor call the function create().<br>
	/// The input is a vector of int and reprezent the topology of perceptron.
	nn_perceptron( std::vector<int>&);
	~nn_perceptron();

	/// Create the perceptron using the layer topology.
	void create( std::vector<int> &);
	void load_from_file( char*);
	/** Forward propagation<br>
	 First argument is the input and the second is the output.**/
	void forward(std::vector<float> &, std::vector<float> &);

	/** This function call forward() and make back propagation<br>
	 First argument is the input and the second is the expected output.**/
	void train( std::vector<float> &, std::vector<float> &);

	///Print ID, in and out for all neurons
	void print();

//private:
	void destroy();

	std::vector<int> topology;
	/// Organizes the neurons
	std::vector<nn_layer> layers;
	/// Neurons-objects
	std::vector<nn_neuron> neurons;
	/// Connections-objects
	std::vector<nn_connection> connection;
	/// Vector of pointers for the input layer
	std::vector<nn_neuron*> inputs_nptr; 
	/// Vector of pointers for the output layer
	std::vector<nn_neuron*> outputs_nptr;

	float learning_speed;

	float error; /// Mean squared error  
	float permitted_error;
	uint64_t status;
};

#endif
