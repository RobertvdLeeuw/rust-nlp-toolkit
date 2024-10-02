use std::ops::{Add, Mul, Sub, Div};

#[derive(Debug)]
struct Matrix {
    data: Vec<Vec<f32>>, // TODO: Enforce rows in cols, or vice versa. (This notation is ambiguous.)
    shape: [usize; 2],  // Rows, cols.
}


impl Matrix {
    fn _create(data: &Vec<Vec<f32>>) -> Self {
        Matrix {
            data: data.clone(),
            shape: [data.len(), data[0].len()]
        }
    }

    fn from(data: Vec<Vec<f32>>) -> Result<Self, String> {
        if !data.iter().all(|row| row.len() == data[0].len()) {
            return Err("Varying row lengths.".to_string());
        }


        Ok(Matrix::_create(&data))
    }

    fn identity(size: usize) -> Matrix {
        let mut data: Vec<Vec<f32>> = vec![vec![0.0; size]; size];

        for i in 0..size {
            data[i][i] = 1.0;
        }

        Matrix::_create(&data)
    }

    fn filled(value: f32, rows: usize, cols: usize) -> Matrix {
        Matrix::_create(&vec![vec![value; rows]; cols])
    }

    fn get_cofactor(&self, i: usize, j: usize) -> f32 {
        let minor_data: Vec<Vec<f32>> = self.data[0..i]
                                            .iter()
                                            .chain(self.data[i+1..].iter())
                                            .map(|row| row[0..j].iter()
                                                                .chain(row[j+1..].iter())
                                                                .cloned()
                                                                .collect()
                                                )
                                            .collect();
        let minor = Matrix::_create(&minor_data);

        f32::powi(-1.0, (i+j) as i32) * minor._calc_determinant()
    }

    fn get_cofactor_matrix(&self) -> Matrix {  // TODO: All this shit as attributes.
        Matrix::_create(&(0..self.shape[0]).map(|i| (0..self.shape[1]).map(|j| self.get_cofactor(i, j))
                                                                      .collect())
                                           .collect())
    }

    fn _calc_determinant(&self) -> f32 {  // TODO: Rewrite without recursion + matrix creation and
                                          // compare speed.
        if self.shape[0] == 2 {
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0];
        }

        (0..self.shape[1]).map(|j| self.get_cofactor(0, j) * self.data[0][j])
                          .sum()
    }

    fn get_determinant(&self) -> Result<f32, String> {
        if self.shape[0] != self.shape[1] {
            return Err(format!("Only square matrices have determinants ({}x{})", self.shape[0], self.shape[1]));
        }

        match self.shape[0] {
            0 => Ok(1.0),
            1 => Ok(self.data[0][0]),
            _ => Ok(self._calc_determinant())
        }
    }

    fn inverse(&self) -> Result<Matrix, String> {
        let det: f32;

        match self.get_determinant() {
            Ok(d) => if d == 0.0 {return Err("Det == 0, so no inverse matrix exists.".to_string())} else {det = d},
            Err(e) => return Err(e)
        }
        
        let adjugate = self.get_cofactor_matrix().transpose();

        Ok(&adjugate / det)
    }

    fn transpose(&self) -> Matrix {
        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut transposed_data: Vec<Vec<f32>> = vec![vec![0.0; rows]; cols];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j][i] = self.data[i][j];
            }
        }

        Matrix::_create(&transposed_data)
    }

    fn print(&self, title: &str) {
        println!("-------- {} ({}x{}) --------", title, self.shape[0], self.shape[1]);

        for row in &self.data {
            println!("{:?}", row);
        }

        println!("-----------------------------");
    }

    fn _perform_matrix_operand<F>(&self, other: &Matrix, operand: F) -> Matrix  // Used for add,
                                                                                 // times, etc.
    where
        F: Fn(&f32, &f32) -> f32, 
    {
        Matrix::_create(&self.data
                             .iter()
                             .zip(other.data.iter())
                             .map(|(s_row, o_row)| {
                                 s_row.iter()
                                      .zip(o_row.iter())
                                      .map(|(s, o)| operand(s, o))
                                      .collect::<Vec<f32>>()
                             })
                             .collect::<Vec<Vec<f32>>>()
                            )
    }

    fn _perform_scalar_operand<F>(&self, scalar: &f32, operand: F) -> Matrix  // Used for add,
                                                                                 // times, etc.
    where
        F: Fn(&f32, &f32) -> f32, 
    {
        Matrix::_create(&self.data
                             .iter()
                             .map(|row| {
                                 row.iter()
                                    .map(|x| operand(x, scalar))
                                    .collect::<Vec<f32>>()
                             })
                             .collect::<Vec<Vec<f32>>>()
                            )
    }
}


impl Add<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn add(self, other: &Matrix) -> Result<Matrix, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Mismatched dimensions: S ({}x{}) - O ({}x{})",
                self.shape[0], self.shape[1], other.shape[0], other.shape[1]
            ));
        }

        Ok(Matrix::_perform_matrix_operand(self, other, |x, y| Add::add(x, y)))
    }
}

impl Add<f32> for &Matrix {
    type Output = Matrix;

    fn add(self, scalar: f32) -> Matrix {
        Matrix::_perform_scalar_operand(self, &scalar, |x, y| Add::add(x, y))
    }
}


impl Sub<&Matrix> for &Matrix {
    type Output = Result<Matrix, String>;

    fn sub(self, other: &Matrix) -> Result<Matrix, String> {
        if self.shape != other.shape {
            return Err(format!(
                "Mismatched dimensions: S ({}x{}) - O ({}x{})",
                self.shape[0], self.shape[1], other.shape[0], other.shape[1]
            ));
        }

        Ok(Matrix::_perform_matrix_operand(self, other, |x, y| Sub::sub(x, y)))
    }
}

impl Sub<f32> for &Matrix {
    type Output = Matrix;

    fn sub(self, scalar: f32) -> Matrix {
        Matrix::_perform_scalar_operand(self, &scalar, |x, y| Sub::sub(x, y))
    }
}


impl Mul<&Matrix> for &Matrix {  // Dot product
    type Output = Result<Matrix, String>;

    fn mul(self, other: &Matrix) -> Result<Matrix, String> {
        if self.shape[1] != other.shape[0] {
            return Err(format!(
                "Mismatched dimensions: S ({}x{}) - O ({}x{})",
                self.shape[0], self.shape[1], other.shape[0], other.shape[1]
            ));
        }

        Ok(Matrix::_create(&self.data
                                .iter()
                                .map(|row_s| {
                                     other.transpose()
                                          .data
                                          .iter()
                                          .map(|row_o| {
                                              row_s.iter()
                                                   .zip(row_o)
                                                   .map(|(x, y)| x*y)
                                                   .sum()
                                          })
                                          .collect()
                                 })
                                .collect()))
    }
}

impl Mul<f32> for &Matrix {
    type Output = Matrix;

    fn mul(self, scalar: f32) -> Matrix {
        Matrix::_perform_scalar_operand(self, &scalar, |x, y| Mul::mul(x, y))
    }
}


impl Div<f32> for &Matrix {
    type Output = Matrix;

    fn div(self, scalar: f32) -> Matrix {
        Matrix::_perform_scalar_operand(self, &scalar, |x, y| Div::div(x, y))
    }
}

// TODO
// Cross product
// Hamarand product

fn main() {
    let matrix1 = Matrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ])
    .expect("");

    matrix1.print("Base 1");

    let matrix2 = Matrix::from(vec![
        vec![1.0, 2.0, 3.0, 9.0],
        vec![4.0, 5.0, 6.0, 12.0],
        vec![3.0, 4.0, 8.0, 7.0],
        vec![2.0, 1.0, 1.0, 1.0],
    ])
    .expect("");

    matrix2.print("Base 2");

    (&matrix1 + &matrix1).expect(""). print("Double");
    (&matrix1 - 3.0).print("Plus 3");

    matrix1.transpose().print("Transposed");

    Matrix::identity(3).transpose().print("Identity");
    Matrix::filled(8.0, 3, 6).transpose().print("Filled");

    (&matrix1 * &matrix1.transpose()).expect("").print("Times");

    println!("DETERMINANT: {}", matrix2.get_determinant().expect(""));
    matrix2.inverse().expect("").print("Inverse");
}


// TODO: N DIMENSION TENSOR: data: Vec<T> where T = Vec<T> or f32
