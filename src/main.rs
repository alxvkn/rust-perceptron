// use std::io::Write;
use std::ops::{AddAssign, Mul};

use iced::Alignment::Center;
use iced::Event;
use iced::widget::{Column, column, text};
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
    FType: Float + AddAssign + Mul<FType, Output = FType> + std::fmt::Display + std::fmt::Debug,
    StandardUniform: Distribution<FType>,
{
    fn new(weights: [[FType; NINPUTS]; NLAYERS], bias: FType) -> Self {
        Perceptron { weights, bias }
    }

    fn activation_function(&self, net_input: FType) -> FType {
        // net_input.max(FType::from(0.).unwrap())
        net_input
    }

    fn fit(&mut self, examples: &[[FType; NINPUTS]], targets: &[FType], iterations: u32) {
        self.bias = rand::random::<FType>();

        for layer in &mut self.weights {
            for weight in layer {
                *weight = rand::random::<FType>();
            }
        }

        'outer: for i in 0..iterations {
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
                    break 'outer;
                }

                for (&input, weight) in example.iter().zip(&mut self.weights[0]) {
                    let learning_rate = 0.01;
                    let weight_correction = error * FType::from(learning_rate).unwrap();
                    *weight += weight_correction * input;
                    self.bias += weight_correction;
                }
            }
        }
        println!(
            "finished training with the following weights {:?}\n",
            self.weights
        );
    }

    fn predict(&self, inputs: [FType; NINPUTS]) -> FType {
        let mut net_input = self.bias;

        for (&input, weight) in inputs.iter().zip(self.weights[0]) {
            net_input += input * weight;
        }

        self.activation_function(net_input)
    }
}

struct Window {
    p: Perceptron<f64, 1, 2>,
    prediction: f64,
    first_value: String,
    second_value: String,
}

#[derive(Debug, Clone)]
pub enum Message {
    // Predict,
    FirstInputChanged(String),
    SecondInputChanged(String),
    Event(iced::Event),
}

impl Default for Window {
    fn default() -> Self {
        let inputs = vec![
            [1., 2.],
            [4., 5.],
            [9., 10.],
            [1., 4.],
            [0.5, 1.],
            [2., 6.],
            [1., 7.],
        ];
        let targets = vec![3., 6., 11., 7., 1.5, 10., 13.];
        let mut p = Perceptron {
            weights: [[0., 0.]],
            bias: 0.,
        };
        p.fit(&inputs, &targets, 1100);

        Window {
            p,
            prediction: 0.,
            first_value: "".to_owned(),
            second_value: "".to_owned(),
        }
    }
}

impl Window {
    pub fn view(&self) -> Column<Message> {
        column![
            iced::widget::text_input("0", &self.first_value).on_input(Message::FirstInputChanged),
            iced::widget::text_input("0", &self.second_value).on_input(Message::SecondInputChanged),
            // We show the value of the counter here
            text(self.prediction).size(50),
        ]
            .align_x(Center)
    }

    pub fn subscription(&self) -> iced::Subscription<Message> {
        iced::event::listen().map(Message::Event)
    }

    pub fn update(&mut self, message: Message) -> iced::Task<Message> {
        match message {
            // Message::Predict => {
            //     self.prediction = self.p.predict([self.first_value.parse().unwrap(), self.second_value.parse().unwrap()]);
            // },
            Message::FirstInputChanged(value) => {
                self.first_value = value;
                self.prediction = self.p.predict([
                    self.first_value.parse().unwrap_or(0.),
                    self.second_value.parse().unwrap_or(0.),
                ]).round();
                iced::Task::none()
            }
            Message::SecondInputChanged(value) => {
                self.second_value = value;
                self.prediction = self.p.predict([
                    self.first_value.parse().unwrap_or(0.),
                    self.second_value.parse().unwrap_or(0.),
                ]).round();
                iced::Task::none()
            }
            Message::Event(event) => match event {
                Event::Keyboard(iced::keyboard::Event::KeyPressed {
                    key: iced::keyboard::Key::Named(iced::keyboard::key::Named::Tab),
                    modifiers,
                    ..
                }) => {
                    if modifiers.shift() {
                        iced::widget::focus_previous()
                    } else {
                        iced::widget::focus_next()
                    }
                }
                _ => iced::Task::none(),
            },
        }
    }
}

fn main() {
    // let mut p = Perceptron::new([[0., 0.]], 0.);
    // let inputs = vec![
    //     [1., 2.],
    //     [4., 5.],
    //     [9., 10.],
    //     [1., 4.],
    //     [0.5, 1.],
    //     [2., 6.],
    //     [1., 7.],
    // ];
    // let targets = vec![
    //     3.,
    //     6.,
    //     11.,
    //     7.,
    //     1.5,
    //     10.,
    //     13.,
    // ];
    // p.fit(
    //     &inputs,
    //     &targets,
    //     1100,
    // );

    let _ = iced::application("hi", Window::update, Window::view)
        .subscription(Window::subscription)
        .run();

    // loop {
    //     print!("enter two numbers of a progression: ");
    //     std::io::stdout().flush().unwrap();
    //     let mut input = String::new();
    //     let _ = std::io::stdin().read_line(&mut input);
    //     let numbers: Result<Vec<f64>, _> = input.split_whitespace().map(|s| s.parse()).collect();
    //
    //     match numbers {
    //         Ok(nums) if nums.len() == 2 => {
    //             println!("prediction for the next element is {}\n", p.predict([nums[0], nums[1]]));
    //         },
    //         _ => panic!()
    //     }
    // }
}
