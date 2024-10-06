use std::iter::{FromIterator, Iterator, Sum};
use std::marker::PhantomData;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

mod td;

use td::TensorData;

#[derive(Clone, Debug)]
pub struct Tensor<const RANK: usize> {
    data: TensorData,
    _phantom: PhantomData<[(); RANK]>,
}


/// Implementation for common tools like indexing, creating from vec, and summing.
impl<const RANK: usize> Index<usize> for Tensor<RANK> {
    type Output = TensorData;
    
    fn index(&self, index: usize) -> &Self::Output {
        match &self.data {
            TensorData::Scalar(_) => panic!("Attempted to index a scalar value"),
            TensorData::NTensor(vec) => &vec[index],
        }
    }
}

impl<const RANK: usize> IndexMut<usize> for Tensor<RANK> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match &mut self.data {
            TensorData::Scalar(_) => panic!("Attempted to index a scalar value"),
            TensorData::NTensor(vec) => &mut vec[index],
        }
    }
}

impl<T, const RANK: usize> FromIterator<T> for Tensor<RANK>
where
    T: Into<TensorData> + Clone,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<TensorData> = iter.into_iter().map(|item| item.into()).collect();

        Tensor {
            data: TensorData::NTensor(vec),
            _phantom: PhantomData,
        }
    }
}

pub enum TensorIter<'a, const RANK: usize> {
    Scalar(std::iter::Once<&'a Tensor<RANK>>),
    NTensor(std::slice::Iter<'a, TensorData>),
}

impl<'a, const RANK: usize> Iterator for TensorIter<'a, RANK> {
    type Item = Tensor<RANK>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TensorIter::Scalar(iter) => iter.next().cloned(),
            TensorIter::NTensor(iter) => iter.next().map(|data| {
                Tensor {
                    data: data.clone(),
                    _phantom: PhantomData,
                }
            }),
        }
    }
}

impl<const RANK: usize> Sum for Tensor<RANK> {
    fn sum<I>(iter: I) -> Tensor<RANK>
    where
        I: Iterator<Item = Tensor<RANK>>,
    {
        Tensor::<RANK> {
            data: iter.fold(TensorData::Scalar(0.0), |acc, x| &x.data + &acc),
            _phantom: PhantomData
        }
    }
}


impl<const RANK: usize> Tensor<RANK> {
    /// Most of this is just a 'frontend' for TensorData, 'real' code and documentation can be
    /// found there.

    pub fn from_td(td: &TensorData) -> Tensor<RANK> {
        Tensor {
            data: td.clone(),
            _phantom: PhantomData
        }
    }

    pub fn iter(&self) -> TensorIter<RANK> {
        match &self.data {
            TensorData::Scalar(_) => TensorIter::Scalar(std::iter::once(self)),
            TensorData::NTensor(vec) => TensorIter::NTensor(vec.iter()),
        }
    }

    pub fn filled(value: f32, shape: Vec<usize>) -> Tensor<RANK> {
        Tensor::<RANK> {
            data: TensorData::filled(value, shape),
            _phantom: PhantomData
        }
    }

    pub fn shape(&self) -> Vec<usize> {
        self.data.shape()
    }

    pub fn shape_as_string(&self) -> String {
        self.data.shape_as_string()
    }

    pub fn len(&self, dim: usize) -> usize {
        self.data.len(dim)  // TODO: flip index logic (make [0] inner and [n] outer.)
    }

    fn perform_operand<F>(&self, other: &Tensor<RANK>, operand: F) -> Tensor<RANK>
    where
        F: Fn(f32, f32) -> f32 + Copy, 
    {
        Tensor::<RANK> {
            data: self.data.perform_operand(&other.data, operand),
            _phantom: PhantomData
        }
    }

    pub fn identity(size: usize, shape: usize) -> Tensor<RANK> {
        Tensor::from_td(&TensorData::identity(size, shape))
    }

    fn transpose(&self, a: usize, b: usize) -> Tensor<RANK> {
        Tensor::from_td(&self.data.transpose(a, b))
    }

    fn full_transpose(&self, new_order: Vec<usize>) -> Tensor<RANK> {
        Tensor::from_td(&self.data.full_transpose(new_order))
    }
    
    fn hamarand_product(&self, other: &Tensor<RANK>) -> Tensor<RANK> {
        Tensor::from_td(&self.data.hamarand_product(&other.data))
    }

}

/* impl<const SELFRANK: usize, const OTHERRANK: usize> Tensor<SELFRANK> {
    pub fn tensor_product(&self, other: &Tensor<OTHERRANK>) -> Tensor<{SELFRANK + OTHERANK - 1}> {
        Tensor::<{SELFRANK + OTHERANK - 1}> {
            data: self.data.tensor_product(&other.data),
            _phantom: PhantomData
        }
    }

    pub fn contract(&self, ranks_to_contract: Vec<usize>) -> Tensor<{RANK - ranks_to_contract.len()}> {
        Tensor::<{RANK - ranks_to_contract.len()}> {
            data: self.data.contract(ranks_to_contract),
            _phantom: PhantomData
        }
    }
} */

// All operators with 2 tensors are Hamarand style
// (except for multiplication, which is the dot product)
macro_rules! impl_tensor_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl<const RANK: usize> $trait for &Tensor<RANK> {
            type Output = Tensor<RANK>;
            fn $method(self, rhs: Self) -> Self::Output {
                self.perform_operand(rhs, $op)
            }
        }

        impl<const RANK: usize> $trait<f32> for &Tensor<RANK> {
            type Output = Tensor<RANK>;
            fn $method(self, rhs: f32) -> Self::Output {
                Tensor::<RANK> {
                    data: self.data.perform_operand(&TensorData::Scalar(rhs), $op),
                    _phantom: PhantomData
                }
            }
        }

        impl<const RANK: usize> $trait<f32> for Tensor<RANK> {
            type Output = Tensor<RANK>;
            fn $method(self, rhs: f32) -> Self::Output {
                Tensor::<RANK> {
                    data: self.data.perform_operand(&TensorData::Scalar(rhs), $op),
                    _phantom: PhantomData
                }
            }
        }

        impl<const RANK: usize> $trait<&Tensor<RANK>> for f32 {
            type Output = Tensor<RANK>;
            fn $method(self, rhs: &Tensor<RANK>) -> Self::Output {
                Tensor::<RANK> {
                    data: rhs.data.perform_operand(&TensorData::Scalar(self), $op),
                    _phantom: PhantomData
                }
            }
        }

        impl<const RANK: usize> $trait<Tensor<RANK>> for f32 {
            type Output = Tensor<RANK>;
            fn $method(self, rhs: Tensor<RANK>) -> Self::Output {
                Tensor::<RANK> {
                    data: rhs.data.perform_operand(&TensorData::Scalar(self), $op),
                    _phantom: PhantomData
                }
            }
        }
    }
}

impl_tensor_op!(Add, add, |a, b| a + b);
impl_tensor_op!(Sub, sub, |a, b| a - b);
impl_tensor_op!(Div, div, |a, b| a / b);

// Special implementation for Mul to handle dot product for Tensor
impl<const RANK: usize> Mul for &Tensor<RANK> {
    type Output = Tensor<RANK>;
    fn mul(self, rhs: Self) -> Self::Output {
        Tensor::<RANK> {
            data: self.data.dot_product(&rhs.data),
            _phantom: PhantomData
        }
    }
}

impl<const RANK: usize> Mul<f32> for &Tensor<RANK> {
    type Output = Tensor<RANK>;
    fn mul(self, rhs: f32) -> Self::Output {
        Tensor::<RANK> {
            data: self.data.perform_operand(&TensorData::Scalar(rhs), |a, b| a * b),
            _phantom: PhantomData
        }
    }
}

impl<const RANK: usize> Mul<f32> for Tensor<RANK> {
    type Output = Tensor<RANK>;
    fn mul(self, rhs: f32) -> Self::Output {
        Tensor::<RANK> {
            data: self.data.perform_operand(&TensorData::Scalar(rhs), |a, b| a * b),
            _phantom: PhantomData
        }
    }
}

impl<const RANK: usize> Mul<&Tensor<RANK>> for f32 {
    type Output = Tensor<RANK>;
    fn mul(self, rhs: &Tensor<RANK>) -> Self::Output {
        Tensor::<RANK> {
            data: rhs.data.perform_operand(&TensorData::Scalar(self), |a, b| a * b),
            _phantom: PhantomData
        }
    }
}

impl<const RANK: usize> Mul<Tensor<RANK>> for f32 {
    type Output = Tensor<RANK>;
    fn mul(self, rhs: Tensor<RANK>) -> Self::Output {
        Tensor::<RANK> {
            data: rhs.data.perform_operand(&TensorData::Scalar(self), |a, b| a * b),
            _phantom: PhantomData
        }
    }
}


// TODO
// Flatten(to dim_n)
// Normalize
    // sum(all): x^2 = 1
// Eigen
// Cosine similarity

// Type aliases for common tensor ranks
pub type Scalar = Tensor<0>;
pub type Vector = Tensor<1>;
pub type Matrix = Tensor<2>;


impl Scalar {
    pub fn from(scalar: f32) -> Scalar {
        Scalar {
            data: TensorData::Scalar(scalar),
            _phantom: PhantomData
        }
    }

    pub fn unwrap(&self) -> f32 {
        match self.data {
            TensorData::Scalar(val) => val,
            TensorData::NTensor(_) => panic!("Scalar has nested data (NTensor)")
        }
    }
}

impl Vector {
    pub fn from(vector: Vec<f32>) -> Vector {
        Vector {
            data: TensorData::NTensor(vector.into_iter()
                                            .map(TensorData::Scalar)
                                            .collect()
            ),
            _phantom: PhantomData
        }
    }

    pub fn cross_product(&self, other: &Vector) -> Vector {
        if self.len(0) != 3 || other.len(0) != 3 {
            panic!("Tried to get the cross product of vectors of wrong length ({}, {}), expected 3.", self.len(0), other.len(0))
        }

        // TODO: automatic casting to f32 to remove .value() everywhere.
        Vector::from(vec![self[1].value()*other[2].value() - self[2].value()*other[1].value(), 
                          self[2].value()*other[0].value() - self[0].value()*other[2].value(), 
                          self[0].value()*other[1].value() - self[1].value()*other[0].value()])
    }

    pub fn print(&self) -> String {
        self.data
            .iter()
            .map(|s| s.value().to_string())
            .collect::<Vec<String>>()
            .join(", ")
    }
}

impl Matrix {
    pub fn from(matrix: Vec<Vec<f32>>) -> Result<Matrix, String> {
        Ok(Matrix {
            data: TensorData::NTensor(matrix.into_iter()
                                            .map(|v| Vector::from(v).data)
                                            .collect()
            ),
            _phantom: PhantomData
        })
    }

     pub fn get_cofactor(&self, p: usize, q: usize) -> f32 {
        let n = self.shape()[0];  // TODO: Back to declarative.
        let mut cofactor = Vec::new();
        for i in 0..n {
            if i == p {
                continue;
            }
            let mut row = Vec::new();
            for j in 0..n {
                if j == q {
                    continue;
                }
                row.push(self.data[i][j].value());
            }
            cofactor.push(row);
        }
        let minor = Matrix::from(cofactor).expect("Impossible error during cofactor matrix creation");
        f32::powi(-1.0, (p + q) as i32) * minor.calc_determinant()
    }

    pub fn calc_determinant(&self) -> f32 {  // TODO: Rewrite without recursion + matrix creation and
                                          // compare speed.
        if self.shape()[0] == 2 {
            return self.data[0][0].value() * self.data[1][1].value() - self.data[0][1].value() * self.data[1][0].value();
        }

        (0..self.shape()[1]).map(|j| (self.get_cofactor(0, j) * self.data[0][j].clone()).value())
                            .sum()
    }

    pub fn get_cofactor_matrix(&self) -> Matrix {  // TODO: All this shit as attributes.
        let data: Vec<TensorData> = (0..self.shape()[0])
            .map(|i| (0..self.shape()[1])
                .map(|j| TensorData::Scalar(self.get_cofactor(i, j)))
                .collect())
            .collect();
        
        Matrix {
            data: TensorData::NTensor(data),
            _phantom: PhantomData
        }
    }

    pub fn get_determinant(&self) -> Result<f32, String> {
        if self.shape()[0] != self.shape()[1] {
            return Err(format!("Only square matrices have determinants ({}x{})", self.shape()[0], self.shape()[1]));
        }

        match self.shape()[0] {
            0 => Ok(1.0),
            1 => Ok(self.data[0][0].value()),
            _ => Ok(self.calc_determinant())
        }
    }

    pub fn inverse(&self) -> Result<Matrix, String> {
        let det: f32;

        match self.get_determinant() {
            Ok(d) => if d == 0.0 {return Err("Det == 0, so no inverse matrix exists.".to_string())} else {det = d},
            Err(e) => return Err(e)
        }
        
        let adjugate = self.get_cofactor_matrix().transpose(0, 1);

        Ok(&adjugate / det)
    }

    pub fn print(&self, title: &str) {
        println!("-------- {} ({}x{}) --------", title, self.shape()[0], self.shape()[1]);

        for row in self.data.iter() {
            println!("| {} |", Vector::from_td(&row).print());
        }

        println!("-----------------------------");
    }
}


fn main() {
    // Test case 1: Basic Matrix operations
    let matrix1 = Matrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]).expect("Failed to create matrix1");

    let matrix2 = Matrix::from(vec![
        vec![7.0, 8.0],
        vec![9.0, 10.0],
        vec![11.0, 12.0],
    ]).expect("Failed to create matrix2");

    matrix1.print("Matrix 1");
    matrix2.print("Matrix 2");

    // Test case 2: Matrix multiplication
    (&matrix1 * &matrix2).print("Matrix 1 * Matrix 2");

    // Test case 3: Element-wise operations
    let matrix3 = Matrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]).expect("Failed to create matrix3");

    (&matrix1 + &matrix3).print("Matrix 1 + Matrix 3 (element-wise addition)");
    (&matrix1 - &matrix3).print("Matrix 1 - Matrix 3 (element-wise subtraction)");
    matrix1.hamarand_product(&matrix3).print("Matrix 1 * Matrix 3 (Hadamard product)");

    // Test case 4: Scalar operations
    (&matrix1 * 2.0).print("Matrix 1 * 2");
    (&matrix1 / 2.0).print("Matrix 1 / 2");

    // Test case 5: Transposition
    matrix1.transpose(0, 1).print("Matrix 1 Transposed");

    // Test case 6: Identity matrix
    Matrix::identity(3, 2).print("3x3 Identity Matrix");

    // Test case 7: Filled matrix
    Matrix::filled(8.0, vec![2, 3]).print("2x3 Matrix filled with 8.0");

    // Test case 8: Determinant and Inverse
    let square_matrix = Matrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
        vec![7.0, 8.0, 9.0],
    ]).expect("Failed to create square_matrix");

    square_matrix.print("Square Matrix");
    println!("Determinant of Square Matrix: {}", square_matrix.get_determinant().expect("Failed to calculate determinant"));

    match square_matrix.inverse() {
        Ok(inv) => inv.print("Inverse of Square Matrix"),
        Err(e) => println!("Error calculating inverse: {}", e),
    }

    // Test case 9: Vector operations
    let vector1 = Vector::from(vec![1.0, 2.0, 3.0]);
    let vector2 = Vector::from(vec![4.0, 5.0, 6.0]);

    println!("Vector 1: {}", vector1.print());
    println!("Vector 2: {}", vector2.print());

    let cross_product = vector1.cross_product(&vector2);
    println!("Cross product of Vector 1 and Vector 2: {}", cross_product.print());

    // Test case 10: Scalar operations
    let scalar1 = Scalar::from(5.0);
    let scalar2 = Scalar::from(3.0);

    println!("Scalar 1: {}", scalar1.unwrap());
    println!("Scalar 2: {}", scalar2.unwrap());
    println!("Scalar 1 + Scalar 2: {}", (&scalar1 + &scalar2).unwrap());
    println!("Scalar 1 * Scalar 2: {}", (&scalar1 * &scalar2).unwrap());

    // Test case 11: Mixed rank operations
    let vector_result = &vector1 * scalar1.unwrap();
    println!("Vector 1 * Scalar 1: {}", vector_result.print());

    let matrix_result = &square_matrix * scalar2.unwrap();
    matrix_result.print("Square Matrix * Scalar 2");

    // Test case 12: Error handling
    let non_square_matrix = Matrix::from(vec![
        vec![1.0, 2.0, 3.0],
        vec![4.0, 5.0, 6.0],
    ]).expect("Failed to create non_square_matrix");

    match non_square_matrix.inverse() {
        Ok(_) => println!("This should not happen"),
        Err(e) => println!("Expected error: {}", e),
    }

    // Test case 13: Full transpose
    let matrix4 = Matrix::from(vec![
        vec![1.0, 2.0, 3.0, 4.0],
        vec![5.0, 6.0, 7.0, 8.0],
        vec![9.0, 10.0, 11.0, 12.0],
    ]).expect("Failed to create matrix4");

    matrix4.print("Matrix 4");
    matrix4.full_transpose(vec![1, 0]).print("Matrix 4 fully transposed");
}
