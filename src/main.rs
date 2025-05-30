use std::ops::{AddAssign, Mul};

use num_traits::Float;
use rand::distr::StandardUniform;
use rand::prelude::Distribution;

struct Perceptron<FType, const NLAYERS: usize, const NINPUTS: usize>
where
    FType: Float,
{
    weights: [[FType; NINPUTS]; NLAYERS], // TODO: layers are not used at all, only the first one
    bias: FType,
}

impl<FType, const NLAYERS: usize, const NINPUTS: usize> Perceptron<FType, NLAYERS, NINPUTS>
where
    FType: Float + AddAssign + Mul<FType, Output = FType> + std::fmt::Display,
    StandardUniform: Distribution<FType>,
{
    fn new(weights: [[FType; NINPUTS]; NLAYERS], bias: FType) -> Self {
        Perceptron {
            weights,
            bias,
        }
    }

    fn activation_function(&self, net_input: FType) -> FType {
        net_input
    }

    fn fit(&mut self, examples: &Vec<[FType; NINPUTS]>, targets: &Vec<FType>, iterations: u32) {
        self.bias = rand::random::<FType>().into();

        for layer in &mut self.weights {
            for weight in layer {
                *weight = rand::random::<FType>().into();
            }
        }

        for i in 0..iterations {
            // std::thread::sleep(std::time::Duration::from_millis(2));

            for (example, &target) in examples.iter().zip(targets) {
                let mut net_input = self.bias;

                for (&input, weight) in example.iter().zip(self.weights[0]) {
                    net_input += input * weight;
                }

                let prediction = self.activation_function(net_input);
                let error = target - prediction;
                println!("error = {error:+}");
                if error == FType::from(0.).unwrap() {
                    println!("got zero error at iteration #{i}");
                    return;
                }

                for (&input, weight) in example.iter().zip(&mut self.weights[0]) {
                    let learning_rate = 0.01;
                    let weight_correction = error * FType::from(learning_rate).unwrap();
                    *weight += weight_correction * input;
                    self.bias += weight_correction;
                }
            }
        }
    }

    fn predict(&self, inputs: [FType; NINPUTS]) -> FType {
        let mut net_input = self.bias;

        for (&input, weight) in inputs.iter().zip(self.weights[0]) {
            net_input += input * weight;
        }

        return self.activation_function(net_input);
    }
}

fn main() {
    let mut p = Perceptron::new([[0., 0.]], 0.);
    let inputs = vec![
        [1., 2.],
        [4., 5.],
        [9., 10.],
        [1., 4.],
        [0.5, 1.],
        [2., 6.],
        [1., 7.],
    ];
    let targets = &vec![
        3.,
        6.,
        11.,
        7.,
        1.5,
        10.,
        13.,
    ];
    p.fit(
        &inputs,
        &targets,
        1100,
    );
    println!("{:?}", p.weights);

    println!("{}", p.predict([8., 16.]));
}
