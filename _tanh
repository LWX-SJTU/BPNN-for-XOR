#include<iostream>
#include<string>
#include<fstream>
#include<math.h>
#include<iomanip>

using namespace std;

double sigmoid(double x);

int main() {
	double TrainIn[4][3] = { {0,0,1},{0,1,1},{1,0,1},{1,1,1} };
	double TrainOut[4] = { 0,1,1,0 };

	double Weight_1[6] = { 0.0543,0.0579,-0.0291,0.0999,1,1 };
	double delta_w1[6] = {};
	double Weight_2[3] = { 0.0801,-0.0605,1 };
	double delta_w2[3] = {};

	double net_hidden[2] = {};
	double val_hidden[2] = {};
	double net_out;
	double val_out;

	double bias = 1.0;

	double lr = 0.2;
	double err_target = 0.008;
	double err = 0.009;
	double err_t = err;

	int it = 0;

	string file = "C://Users//as928//Desktop//DATA.txt";
	fstream data;
	data.open(file);

	/*
  //learn rate experiment
	while (lr < 0.9) {

		lr += 0.02;
		err = 0.009;
		it = 0;
		//init
		double Weight_1[6] = { 0.0543,0.0579,-0.0291,0.0999,1,1 };
		double delta_w1[6] = {};
		double Weight_2[3] = { 0.0801,-0.0605,1 };
		double delta_w2[3] = {};

		double net_hidden[2] = {};
		double val_hidden[2] = {};
		double net_out;
		double val_out;

		double bias = 1.0;
  */
    
		while (err > err_target) {
			//err = 0;
			it++;
			err = 0;
			for (int i = 0; i < 4; i++) {
				net_hidden[0] = TrainIn[i][0] * Weight_1[0] + TrainIn[i][1] * Weight_1[2] + bias * Weight_1[4];
				val_hidden[0] = tanh(net_hidden[0]);
				net_hidden[1] = TrainIn[i][0] * Weight_1[1] + TrainIn[i][1] * Weight_1[3] + bias * Weight_1[5];
				val_hidden[1] = tanh(net_hidden[1]);
				net_out = val_hidden[0] * Weight_2[0] + val_hidden[1] * Weight_2[1] + bias * Weight_2[2];
				val_out = tanh(net_out);

				err_t = 0.5 * (TrainOut[i] - val_out) * (TrainOut[i] - val_out);
				if (err_t > err)err = err_t;

				delta_w2[0] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * val_hidden[0];
				delta_w2[1] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * val_hidden[1];
				delta_w2[2] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * bias;

				delta_w1[0] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * Weight_2[0] * (1 - val_hidden[0] * val_hidden[0]) * TrainIn[i][0];
				delta_w1[1] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * Weight_2[1] * (1 - val_hidden[1] * val_hidden[1]) * TrainIn[i][0];
				delta_w1[2] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * Weight_2[0] * (1 - val_hidden[0] * val_hidden[0]) * TrainIn[i][1];
				delta_w1[3] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * Weight_2[1] * (1 - val_hidden[1] * val_hidden[1]) * TrainIn[i][1];
				delta_w1[4] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * Weight_2[0] * (1 - val_hidden[0] * val_hidden[0]) * bias;
				delta_w1[5] = (TrainOut[i] - val_out) * (1 - val_out * val_out) * Weight_2[1] * (1 - val_hidden[1] * val_hidden[1]) * bias;

				Weight_2[0] += delta_w2[0] * lr;
				Weight_2[1] += delta_w2[1] * lr;
				Weight_2[2] += delta_w2[2] * lr;

				Weight_1[0] += delta_w1[0] * lr;
				Weight_1[1] += delta_w1[1] * lr;
				Weight_1[2] += delta_w1[2] * lr;
				Weight_1[3] += delta_w1[3] * lr;
				Weight_1[4] += delta_w1[4] * lr;
				Weight_1[5] += delta_w1[5] * lr;

			}
			
			if (it % 10 == 0) {
		    //
				data << it << " ";
				data << err << endl;
				//system("pause");
			}
      
			//report every 1000 times
			//if (it % 1000 == 999)cout << endl << it << endl;
		}

		cout << "学习率：  " << lr << endl;
		cout << "迭代次数:   " << it << endl;
		//data << lr << " " << it << endl;
		cout << endl;
    
  //when lr experiment,remember this }
	//}
  
	data.close();
	
	//cout << "pred:::" << endl;
	

	system("pause");
	for (int i = 0; i < 4; i++) {
		net_hidden[0] = TrainIn[i][0] * Weight_1[0] + TrainIn[i][1] * Weight_1[2] + bias * Weight_1[4];
		val_hidden[0] = tanh(net_hidden[0]);
		net_hidden[1] = TrainIn[i][0] * Weight_1[1] + TrainIn[i][1] * Weight_1[3] + bias * Weight_1[5];
		val_hidden[1] = tanh(net_hidden[1]);
		net_out = val_hidden[0] * Weight_2[0] + val_hidden[1] * Weight_2[1] + bias * Weight_2[2];
		val_out = tanh(net_out);

		err_t = 0.5 * (TrainOut[i] - val_out) * (TrainOut[i] - val_out);
		cout << endl;
		cout << Weight_1[4] << " " << Weight_1[5] << " " << Weight_2[2] << endl;
		cout << "in 1:    " << TrainIn[i][0] << "    in 2:   "<<TrainIn[i][1] << endl;
		cout << "out :    " << TrainOut[i] << "    real out:   " << val_out << endl;
		cout << "err:   " << err_t << endl;
	}
	system("pause");
}

double sigmoid(double x) {
	return 1 / (1 + exp(-x));
}
