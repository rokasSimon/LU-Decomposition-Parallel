use ndarray::prelude::*;
use rayon::prelude::*;
use std::time::{Instant};
use ndarray::Zip;

use LP_IP::matrix_generation::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        println!("Usage:");
        println!("  run --base_test_file_path --n_tests --thread_count");
        println!("  gen --base_test_file_path --matrix_sizes..");
        return
    }

    if args.len() < 4 {
        println!("Too few arguments!");
        return
    }

    let action = &args[1];
    if action == "run" {
        let base_path = &args[2];
        let n_tests: usize = args[3].parse().unwrap();
        let thread_count: usize = args[4].parse().unwrap();
        println!("[Threads: {}]", thread_count);
        rayon::ThreadPoolBuilder::new().num_threads(thread_count).build_global().unwrap();

        run_tests(base_path, n_tests, thread_count);
    } else if action == "gen" {
        let base_path = &args[2];
        let ndx: Vec<u32> = args[3..].iter().map(|arg| arg.parse().unwrap()).collect();

        generate_test_files(base_path, &ndx);
    } else {
        println!("Unknown command: {}", action);
    }
}

fn generate_test_file(path: &str, n: u32) {
    let res = generate_matrix_to_file(path, n);

    match res {
        Err(e) => {
            println!("{}", e);
            return;
        },
        _ => ()
    }
}

fn generate_test_files(base_path: &str, ndx: &Vec<u32>) {
    for (i, n) in ndx.iter().enumerate() {
        generate_test_file(&format!("{}{}.txt", base_path, i + 1), *n);
    }
}

fn run_tests(base_path: &str, n: usize, thread_count: usize) {
    for i in 0..n {
        let test_file_path = format!("{}{}.txt", base_path, i + 1);
        let res = deserialize_matrix(&test_file_path);

        println!("[File: {}]", i + 1);
        if let Ok((a, b, x)) = res {
            if thread_count == 1 {
                let start = Instant::now();
                let result = lu_decomposition(&a, &b);
                let duration = start.elapsed();

                let result= flatten_array(result);
                let expected = flatten_mat2(x);
                let correct = check_answer(&result, &expected);

                if !correct {
                    println!("{}", result);
                    println!("{}", expected);
                    panic!("{}-th test doesn't solve correctly!", i + 1);
                } else {
                    println!("  [Duration: {} ms]", duration.as_millis());
                }
            } else {
                let start = Instant::now();
                let result = parallel_lu_decomposition(&a, &b);
                let duration = start.elapsed();

                let result= flatten_array(result);
                let expected = flatten_mat2(x);
                let correct = check_answer(&result, &expected);

                if !correct {
                    println!("{}", result);
                    println!("{}", expected);
                    panic!("{}-th test doesn't solve correctly!", i);
                } else {
                    println!("  [Duration: {} ms]", duration.as_millis());
                }
            }
        } else {
            panic!("Unforeseen error on {}-th test!", i);
        }
    }
}

fn check_answer(result: &Array1<i32>, expected: &Array1<i32>) -> bool {
    Zip::from(result)
        .and(expected)
        .all(|&a, &b| a == b)
}

fn lu_decomposition(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    let n = a.shape()[0];

    let mut l = Array2::<f64>::eye(n);
    let mut u = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in i..n {
            //let sum = l.slice(s![i, 0..i]).dot(&u.slice(s![0..i, j]));
            let mut sum = 0.;

            for k in 0..i {
                sum = sum + l[[i, k]] * u[[k, j]];
            }

            u[[i, j]] = a[[i, j]] - sum;
        }
        for j in i+1..n {
            //let sum = l.slice(s![j, 0..i]).dot(&u.slice(s![0..i, i]));
            let mut sum = 0.;

            for k in 0..i {
                sum = sum + l[[j, k]] * u[[k, i]];
            }

            l[[j, i]] = (1. / u[[i, i]]) * (a[[j, i]] - sum);
        }
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let sum = l.slice(s![i, 0..i]).dot(&y.slice(s![0..i]));

        y[[i]] = b[[i, 0]] - sum;
    }

    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let sum = u.slice(s![i, i+1..n]).dot(&x.slice(s![i+1..n]));

        x[[i]] = (1. / u[[i, i]]) * (y[[i]] - sum);
    }

    return x;
}

fn parallel_lu_decomposition(a: &Array2<f64>, b: &Array2<f64>) -> Array1<f64> {
    let n = a.shape()[0];

    let mut l = Array2::<f64>::eye(n);
    let mut u = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        let cols: Vec<(usize, f64)> = (i..n).into_par_iter().map(|j| {
            let sum = l.slice(s![i, 0..i]).dot(&u.slice(s![0..i, j]));

            (j, a[[i, j]] - sum)
        }).collect();

        for (j, au) in cols {
            u[[i, j]] = au;
        }

        let rows: Vec<(usize, f64)> = (i+1..n).into_par_iter().map(|j| {
           let sum = l.slice(s![j, 0..i]).dot(&u.slice(s![0..i, i]));

            (j, (1. / u[[i, i]]) * (a[[j, i]] - sum))
        }).collect();

        for (j, al) in rows {
            l[[j, i]] = al;
        }
    }

    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let sum = l.slice(s![i, 0..i]).dot(&y.slice(s![0..i]));

        y[[i]] = b[[i, 0]] - sum;
    }

    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let sum = u.slice(s![i, i+1..n]).dot(&x.slice(s![i+1..n]));

        x[[i]] = (1. / u[[i, i]]) * (y[[i]] - sum);
    }

    return x;
}