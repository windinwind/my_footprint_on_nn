#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define IN_DIM 2
#define HIDDEN_DIM 4
#define OUT_DIM 2

double sigmoid(double x) {
	double result = 1.0 / (1.0 + exp(0 - x));
	return result;
}

void printArr(double data[], int size) {
	for (int i = 0; i < size; i++) {
		printf("%lf ", data[i]);
	}
	printf("\n");
}

void printArrInInt(double data[], int size) {
	for (int i = 0; i < size; i++) {
		printf("%d ", (int)(data[i] + 0.5));
	}
	printf("\n");
}

int main(void) {
	double Wih[IN_DIM][HIDDEN_DIM]; // Weight matrix of input layer and hidden layer
	double Who[HIDDEN_DIM][OUT_DIM]; // Weight matrix of hidden layer and output layer

	// Initialize Wih
	for (int i = 0; i < IN_DIM; i++) {
		for (int j = 0; j < HIDDEN_DIM; j++) {
			Wih[i][j] = rand() / (double)RAND_MAX;
		}
	}

	// Initialize Who
	for (int i = 0; i < HIDDEN_DIM; i++) {
		for (int j = 0; j < OUT_DIM; j++) {
			Who[i][j] = rand() / (double)RAND_MAX;
		}
	}

	double learning_rate = 0.5;
	double E1 = 100000.0;
	double E2 = E1;
	int count = 0;

	while (1) {
		int print_data = 0;

		if (count == 0) {
			print_data = 1;
		}

		count += 1;

		if (count == 100000) {
			count = 0;
		}

		for (int in_data = 0; in_data < 4; in_data++) {
			double in[IN_DIM];
			double h[HIDDEN_DIM];
			double out[OUT_DIM];

			// Get training data
			in[0] = (double)(in_data & 1);
			in[1] = (double)((in_data & 2) >> 1);

			// Expected output
			double expected_out[OUT_DIM];
			expected_out[0] = (double)((in_data & 1) ^ ((in_data & 2) >> 1));
			expected_out[1] = (double)((in_data & 1) & ((in_data & 2) >> 1));

			// Feed forward
			// Calculate hidden layer output
			for (int i = 0; i < HIDDEN_DIM; i++) {
				// Calculate net
				double net = 0.0;
				for (int j = 0; j < IN_DIM; j++) {
					net += Wih[j][i] * in[j];
				}

				// Calculate sigmoid(net)
				h[i] = sigmoid(net);
			}

			// Calculate output layer output
			for (int i = 0; i < OUT_DIM; i++) {
				// Calculate net
				double net = 0.0;
				for (int j = 0; j < HIDDEN_DIM; j++) {
					net += Who[j][i] * h[j];
				}

				// Calculate sigmoid(net)
				out[i] = sigmoid(net);
			}

			// Calculate error
			double E = 0.0;
			for (int i = 0; i < OUT_DIM; i++) {
				double temp = (expected_out[i] - out[i]);
				E += temp * temp / 2.0;
			}

			double updated_Who[HIDDEN_DIM][OUT_DIM];
			// Back Propagation
			double dE_dnet[OUT_DIM];
			// Update Who
			for (int i = 0; i < OUT_DIM; i++) {
				double dE_dout = -(expected_out[i] - out[i]);
				double dout_dnet = out[i] * (1.0 - out[i]);
				dE_dnet[i] = dE_dout * dout_dnet;

				for (int j = 0; j < HIDDEN_DIM; j++) {
					double dnet_dw = h[j];
					double dE_dw = dE_dout * dout_dnet * dnet_dw;

					// Calculate updated weight matrix
					updated_Who[j][i] = Who[j][i] - learning_rate * dE_dw;
				}
			}

			// Update Wih
			// Calculate dE/dh first
			double dE_dh[HIDDEN_DIM];
			for (int i = 0; i < HIDDEN_DIM; i++) {
				dE_dh[i] = 0.0;
				for (int j = 0; j < OUT_DIM; j++) {
					dE_dh[i] += dE_dnet[j] * Who[i][j];
				}
			}

			double updated_Wih[IN_DIM][HIDDEN_DIM];
			for (int i = 0; i < HIDDEN_DIM; i++) {
				double dh_dnet = h[i] * (1 - h[i]);

				for (int j = 0; j < IN_DIM; j++) {
					double dnet_dw = in[j];
					double dE_dw = dE_dh[i] * dh_dnet * dnet_dw;

					updated_Wih[j][i] = Wih[j][i] - learning_rate * dE_dw;
				}
			}

			// Update weight matrices
			for (int i = 0; i < IN_DIM; i++) {
				for (int j = 0; j < HIDDEN_DIM; j++) {
					Wih[i][j] = updated_Wih[i][j];
					for (int k = 0; k < OUT_DIM; k++) {
						Who[i][j] = updated_Who[i][j];
					}
				}
			}

			if (print_data) {
				printf("\nin data:\n");
				printArr(in, IN_DIM);
				printf("\nout data:\n");
				printArr(out, OUT_DIM);
				printf("\nconverted out data:\n");
				printf("%d%d\n", (int)(out[1] + 0.5), (int)(out[0] + 0.5));
				//printArrInInt(out, OUT_DIM);

				printf("Wih[0]: ");
				printArr(Wih[0], HIDDEN_DIM);
				printf("Wih[1]: ");
				printArr(Wih[1], HIDDEN_DIM);

				for (int i = 0; i < HIDDEN_DIM; i++) {
					printf("Who[%d]: ", i);
					printArr(Who[i], OUT_DIM);
				}
				system("pause");
			}
		}
	}

	system("pause");
	return 0;
}