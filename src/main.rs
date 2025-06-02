#![windows_subsystem = "windows"]
use std::fmt::{Debug, Display};
use std::io::Write;
use std::iter::Sum;
use std::ops::{AddAssign, Mul};

use iced::Alignment::Center;
use iced::Event;
use iced::widget::{Column, column, text};
use num_traits::Float;
use rand::distr::StandardUniform;
use rand::prelude::Distribution;
use rand::random;

struct Perceptron<FType, const NLAYERS: usize, const NINPUTS: usize>
where
    FType: Float,
{
    weights: [[FType; NINPUTS]; NLAYERS], // TODO: layers are not used at all, only the first one
    bias: FType,
}

impl<FType, const NLAYERS: usize, const NINPUTS: usize> Perceptron<FType, NLAYERS, NINPUTS>
where
    FType: Float + AddAssign + Mul<FType, Output = FType> + Display + Debug + Sum,
    StandardUniform: Distribution<FType>,
{
    fn activation_function(&self, net_input: FType) -> FType {
        // net_input.max(FType::from(0.).unwrap())
        net_input
    }

    fn fit(&mut self, examples: &[[FType; NINPUTS]], targets: &[FType], iterations: u32) {
        self.bias = random::<FType>();

        self.weights.iter_mut().for_each(|layer| {
            layer
                .iter_mut()
                .for_each(|weight| *weight = random::<FType>())
        });

        'outer: for i in 0..iterations {
            // std::thread::sleep(std::time::Duration::from_millis(2));

            for (example, &target) in examples.iter().zip(targets) {
                let prediction = self.predict(example);
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

    fn predict(&self, inputs: &[FType; NINPUTS]) -> FType {
        self.activation_function(
            self.bias
                + inputs
                    .iter()
                    .zip(self.weights[0])
                    .map(|(&input, weight)| input * weight)
                    .sum(),
        )
    }
}

struct Window {
    p: Perceptron<f64, 1, 2>,
    prediction: String,
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
            prediction: "".to_owned(),
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
            text(&self.prediction).size(50),
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
                self.prediction = self
                    .p
                    .predict(&[
                        self.first_value.parse().unwrap_or(0.),
                        self.second_value.parse().unwrap_or(0.),
                    ])
                    .round()
                    .to_string();
                iced::Task::none()
            }
            Message::SecondInputChanged(value) => {
                self.second_value = value;
                self.prediction = self
                    .p
                    .predict(&[
                        self.first_value.parse().unwrap_or(0.),
                        self.second_value.parse().unwrap_or(0.),
                    ])
                    .round()
                    .to_string();
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

fn iced_main() {
    let _ = iced::application(
        |w: &Window| {
            let strings = [&w.first_value, &w.second_value, &w.prediction];
            strings
                .iter()
                .map(|s| if s.is_empty() { "0" } else { s })
                .collect::<Vec<&str>>()
                .join(", ")
        },
        Window::update,
        Window::view,
    )
    .subscription(Window::subscription)
    .window(iced::window::Settings {
        size: iced::Size {
            width: 300.,
            height: 300.,
        },
        ..Default::default()
    })
    .run();
}

fn cli_main() {
    let mut p = Perceptron {
        weights: [[0., 0.]],
        bias: 0.,
    };
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
    p.fit(&inputs, &targets, 1100);

    loop {
        print!("enter two numbers of a progression: ");
        std::io::stdout().flush().unwrap();
        let mut input = String::new();
        let _ = std::io::stdin().read_line(&mut input);
        let numbers: Result<Vec<f64>, _> = input.split_whitespace().map(|s| s.parse()).collect();

        match numbers {
            Ok(nums) if nums.len() == 2 => {
                println!(
                    "prediction for the next element is {}\n",
                    p.predict(&[nums[0], nums[1]])
                );
            }
            _ => panic!(),
        }
    }
}

fn main() {
    if std::env::args().next_back() == Some("--cli".into()) {
        cli_main()
    } else {
        iced_main();
    }
}
