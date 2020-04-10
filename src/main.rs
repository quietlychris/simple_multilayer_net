extern crate ndarray as nd;
extern crate ndarray_linalg as ndl;
extern crate ndarray_rand as ndr;
extern crate ndarray_stats as nds;

use nd::prelude::*;


fn main() {
    let x: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let (n, m) = (x.shape()[0], x.shape()[1]);
    let output: Array2<f32> = array![[1.], [0.], [1.]];
    let mut w1: Array2<f32> = array![[0.7, 0.28], [0.32, 0.35], [0.95, 0.05], [0.78, 0.83]];
    let mut w2: Array2<f32> = array![[0.63], [0.59]];
    println!("w1:\n{:#?}", w1);
    println!("w2:\n{:#?}", w2);

    println!("x start:\n{:#?}", x);
    let learnrate = 0.1;

    for i in 0..10000 {
        // Forward pass
        let a1 = x.clone();
        let z2 = a1.dot(&w1);
        //println!("z2 start:\n{:#?}",z2);
        let a2 = z2.mapv(|x| sigmoid(x));
        //println!("a2 (z2 after sigmoid):\n{:#?}",a2);
        let z3 = a2.dot(&w2);
        //println!("z3 start:\n{:#?}",z3);
        let a3 = z3.mapv(|x| sigmoid(x));
        //println!("a3 (z3 after sigmoid):\n{:#?}",a3);

        let error = (&a3 - &output).sum();
        println!("error #{}: {}", i, error);

        // Backwards pass
        let delta3 = (a3 - output.clone()) * z3.mapv(|x| sigmoid_prime(x)) * learnrate;
        //println!("delta3:\n{:#?}",delta3);
        let delta_w2 = a2.t().dot(&delta3);
        w2 = w2 - delta_w2;

        let delta2 = delta3.dot(&w2.t()) * z3.clone().mapv(|x| sigmoid_prime(x));
        let delta_w1 = a1.t().dot(&delta2);
        w1 = w1 - delta_w1;
    }

    let test_x: Array2<f32> = array![[4., 2., 2., 1.], [1., 1., 3., 4.]];

    // Forward pass on test data with new weights
    let a1 = test_x.clone();
    let z2 = a1.dot(&w1);
    //println!("z2 start:\n{:#?}",z2);
    let a2 = z2.mapv(|x| sigmoid(x));
    //println!("a2 (z2 after sigmoid):\n{:#?}",a2);
    let z3 = a2.dot(&w2);
    //println!("z3 start:\n{:#?}",z3);
    let a3 = z3.mapv(|x| sigmoid(x));
    let test_output: Array2<f32> = array![[0.], [1.]];
    println!("The result should be close to the presumed output of {:#?}",test_output);
    println!("result: {:#?}", a3);
}



fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
