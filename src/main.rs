extern crate ndarray as nd;
extern crate ndarray_linalg as ndl;
extern crate ndarray_rand as ndr;
extern crate ndarray_stats as nds;

use nd::prelude::*;


fn main() {
    let x: Array2<f32> = array![[1., 2., 3., 4.], [4., 3., 2., 1.], [1., 2., 2.5, 4.]];
    let (n, m) = (x.shape()[0], x.shape()[1]);
    let output: Array2<f32> = array![[1.0,0.0], [0.,1.0], [1.0,0.0]];
    let mut w0: Array2<f32> = array![[0.7, 0.28], [0.32, 0.35], [0.95, 0.05], [0.78, 0.83]];
    let mut w1: Array2<f32> = array![[0.63, 0.43], [0.59,0.33]];

    let mut b1: Array2<f32> = Array::zeros((3,2));
    let mut b2: Array2<f32> = Array::zeros((3,2));

    // let mut b2: Array2<f32> = array![[0.01,-0.01],[0.01,-0.01],[0.01,-0.01]];

    println!("w0:\n{:#?}", w0);
    println!("w1:\n{:#?}", w1);

    println!("x start:\n{:#?}", x);
    let learnrate = 0.1;
    let bias_learnrate = 0.03;

    for i in 0..10000 {
        // Forward pass
        let a0 = x.clone();
        let z1 = a0.dot(&w0);
        //println!("z1 start:\n{:#?}",z1);
        let a1 = z1.mapv(|x| sigmoid(x)) + &b1;
        // println!("a1 is of shape {:?}",a1.shape());
        //println!("a1 (z1 after sigmoid):\n{:#?}",a1);
        let z2 = a1.dot(&w1);
        //println!("z2 start:\n{:#?}",z2);
        let a2 = z2.mapv(|x| sigmoid(x)) + &b2;
        //println!("a2 (z2 after sigmoid):\n{:#?}",a2);
        let error = (&a2 - &output).sum();
        println!("error #{}: {}", i, error);

        // Backwards pass
        let delta2 = (&a2 - &output.clone()) * z2.mapv(|x| sigmoid_prime(x)) * learnrate;
        // println!("delta2 is of shape: {:?}",delta2.shape());
        //println!("delta2:\n{:#?}",delta2);
        let delta_w1 = a1.t().dot(&delta2);
        w1 = w1 - delta_w1;
        b2 = b2 + (&delta2 * bias_learnrate);

        let delta1 = delta2.dot(&w1.t()) * z1.clone().mapv(|x| sigmoid_prime(x));
        let delta_w0 = a0.t().dot(&delta1);
        w0 = w0 - delta_w0;
        b1 = b1 + (&delta1 * bias_learnrate);
        //println!("b2:\n{:#?}",b2);
        //println!("b1:\n{:#?}",b1);

    }

    let test_x: Array2<f32> = array![[4., 2., 2., 1.], [1., 1., 3., 4.]];

    // Forward pass on test data with new weights
    let a0 = test_x.clone();
    let z1 = a0.dot(&w0);
    //println!("z2 start:\n{:#?}",z2);
    let a1 = z1.mapv(|x| sigmoid(x));
    //println!("a2 (z2 after sigmoid):\n{:#?}",a2);
    let z2 = a1.dot(&w1);
    //println!("z3 start:\n{:#?}",z3);
    let a2 = z2.mapv(|x| sigmoid(x));
    let test_output: Array2<f32> = array![[0.0,1.0], [1.0,0.0]];
    println!("The result should be close to the presumed output of\n{:#?}",test_output);
    println!("result:\n{:#?}", a2);
}



fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_prime(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}
