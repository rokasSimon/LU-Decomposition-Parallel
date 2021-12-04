pub mod matrix_generation {
    use std::fs::File;
    use std::io::{BufRead, BufReader, BufWriter, Error, Write};
    use ndarray::prelude::*;

    pub fn deserialize_matrix(path: &str) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>), Error> {
        let file = File::open(path)?;

        let lines: Vec<String> = BufReader::new(file)
            .lines()
            .map(|line| line.expect("Could not parse line."))
            .collect();

        let n: usize = lines[0].parse().unwrap();
        let a: Vec<f64> = lines[1]
            .split_whitespace()
            .map(|num| num.parse().unwrap())
            .collect();
        let b: Vec<f64> = lines[2]
            .split_whitespace()
            .map(|num| num.parse().unwrap())
            .collect();
        let x: Vec<f64> = lines[3]
            .split_whitespace()
            .map(|num| num.parse().unwrap())
            .collect();

        let a: Array2<f64> = Array2::from_shape_vec((n, n), a).unwrap();
        let b: Array2<f64> = Array2::from_shape_vec((n, 1), b).unwrap();
        let x: Array2<f64> = Array2::from_shape_vec((n, 1), x).unwrap();

        Ok((a, b, x))
    }

    pub fn serialize_matrix(path: &str, a: &Array2<f64>, b: &Array2<f64>, x: &Array2<f64>) -> Result<(), Error> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        let n = a.shape()[0];
        let a = Array::from_iter(a.into_iter())
            .map(|&f| f.to_string())
            .to_vec()
            .join(" ");
        let b = Array::from_iter(b.into_iter())
            .map(|&f| f.to_string())
            .to_vec()
            .join(" ");
        let x = Array::from_iter(x.into_iter())
            .map(|f| (f.round() as i32).to_string())
            .to_vec()
            .join(" ");

        writer.write_fmt(format_args!("{}\n", n))?;
        writer.write_fmt(format_args!("{}\n", a))?;
        writer.write_fmt(format_args!("{}\n", b))?;
        writer.write_fmt(format_args!("{}\n", x))?;

        Ok(())
    }

    pub fn generate_random_solvable_system(n: usize) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        use rand::Rng;

        const L_RANGE: i32 = -30;
        const U_RANGE: i32 = 30;

        let mut rng = rand::thread_rng();

        let mut x = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let int = rng.gen_range(L_RANGE..U_RANGE);
            x[[i, 0]] = int as f64;
        }

        let mut b = Array2::<f64>::zeros((n, 1));
        for i in 0..n {
            let int = rng.gen_range(L_RANGE..U_RANGE);
            b[[i, 0]] = int as f64;
        }

        let mut a = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let int = rng.gen_range(L_RANGE..U_RANGE);
                a[[i, j]] = int as f64;
            }

            let mult: f64 = a.row(i).dot(&x.column(0));
            if mult < b[[i, 0]] {
                let dif = b[[i, 0]] - mult;
                let mut rand_idx = rng.gen_range(0..n-1);

                while x[[rand_idx, 0]] == 0. {
                    rand_idx = rng.gen_range(0..n-1);
                }
                let div = dif / x[[rand_idx, 0]];

                a[[i, rand_idx]] = a[[i, rand_idx]] + div;
            } else if mult > b[[i, 0]] {
                let dif = mult - b[[i, 0]];
                let mut rand_idx = rng.gen_range(0..n-1);

                while x[[rand_idx, 0]] == 0. {
                    rand_idx = rng.gen_range(0..n-1);
                }
                let div = dif / x[[rand_idx, 0]];

                a[[i, rand_idx]] = a[[i, rand_idx]] - div;
            }
        }

        return (a, b, x);
    }

    pub fn generate_matrix_to_file(path: &str, n: u32) -> Result<(), Error> {
        let (a, b, x) = generate_random_solvable_system(n as usize);

        serialize_matrix(path, &a, &b, &x)?;

        Ok(())
    }

    pub fn flatten_mat2(mat: Array2<f64>) -> Array1<i32> {
        Array::from_iter(mat.into_iter()).map(|f| f.round() as i32)
    }

    pub fn flatten_array(mat: Array1<f64>) -> Array1<i32> {
        mat.map(|f| f.round() as i32)
    }
}