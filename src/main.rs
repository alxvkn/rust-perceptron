use num_traits::Float;

struct Perceptron<FType, const NLAYERS: usize, const NINPUTS: usize> where FType: Float {
    weights: [[FType; NINPUTS]; NLAYERS], // TODO: layers are not used at all, only the first one
    bias: FType
}

impl<FType, const NLAYERS: usize, const NINPUTS: usize> Perceptron<FType, NLAYERS, NINPUTS>
where FType: From<f64> + Float + std::ops::AddAssign + std::ops::Mul {
    fn new(weights: [[FType; NINPUTS]; NLAYERS], bias: FType) -> Self {
        return Perceptron {
            weights: weights,
            bias: bias,
        }
    }

    fn activation_function(&self, net_input: FType) -> FType {
        net_input
    }

    fn fit(&mut self, inputs: Vec<[FType; NINPUTS]>, targets: Vec<FType>, iterations: u32) {
        self.bias = rand::random::<f64>().into();

        for _ in 0..iterations {
            for inputs_index in 0..inputs.len() {
                let mut net_input = self.bias;
                for i in 0..NINPUTS {
                    net_input += inputs[inputs_index][i] * self.weights[0][i];
                }

                let prediction = self.activation_function(net_input);

                for i in 0..NINPUTS {
                    let weight_correction = (targets[inputs_index] - prediction) * 0.01.into();
                    self.weights[0][i] += weight_correction * inputs[inputs_index][i];
                    self.bias += weight_correction;
                }
            }
        }
    }

    fn predict(&self, inputs: [FType; NINPUTS]) -> FType {
        let mut net_input = self.bias;

        for i in 0..inputs.len() {
            net_input += inputs[i] * self.weights[0][i];
        }

        return self.activation_function(net_input);
    }
}

fn main() {
    let mut p = Perceptron::<f64, 1, 3>::new([[0., 0., 0.]], 0.);
    p.fit(vec![
        [2., 3., 4.],
        [1., 2., 3.],
        [8., 9., 10.],
        [4., 7., 10.],
        [1., 6., 11.],
        [2., 4., 8.],
    ],
        vec![
            5.,
            4.,
            11.,
            13.,
            16.,
            16.,
        ], 1000);
    println!("{:?}", p.weights);

    println!("{}", p.predict([
        8.,
        16.,
        24.,
    ]));
}
