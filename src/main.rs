use std::collections::HashMap;
use std::iter::{FromIterator, Iterator, Sum};
use std::marker::PhantomData;
use std::ops::{Add, Div, Index, IndexMut, Mul, Sub};

#[derive(Clone, Debug)]
pub enum TensorData {
    Scalar(f32),
    NTensor(Vec<TensorData>),
}


impl Index<usize> for TensorData {
    type Output = TensorData;
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            TensorData::Scalar(_) => panic!("Attempted to index a scalar value"),
            TensorData::NTensor(vec) => &vec[index],
        }
    }
}

impl IndexMut<usize> for TensorData {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            TensorData::Scalar(_) => panic!("Attempted to index a scalar value"),
            TensorData::NTensor(vec) => &mut vec[index],
        }
    }
}

impl FromIterator<TensorData> for TensorData {
    fn from_iter<I: IntoIterator<Item = TensorData>>(iter: I) -> Self {
        TensorData::NTensor(iter.into_iter().collect())
    }
}

enum TensorDataIter<'a> {
    Scalar(std::iter::Once<&'a TensorData>),
    NTensor(std::slice::Iter<'a, TensorData>),
}

impl<'a> Iterator for TensorDataIter<'a> {
    type Item = TensorData;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            TensorDataIter::Scalar(iter) => iter.next().cloned(),
            TensorDataIter::NTensor(iter) => iter.next().cloned(),
        }
    }
}

impl Sum for TensorData {
    fn sum<I>(iter: I) -> TensorData
    where
        I: Iterator<Item = TensorData>,
    {
        iter.fold(TensorData::Scalar(0.0), |acc, x| {
            match (&acc, &x) {
                (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(a + b),
                _ => panic!("Cannot sum tensors of different ranks"),
            }
        })
    }
}


impl TensorData {
    fn shape(&self) -> Vec<usize> {
        match self {
            TensorData::Scalar(_) => vec![],
            TensorData::NTensor(vec) => {
                let mut s = vec![vec.len()];
                s.extend(vec[0].shape());

                s
            }
        }
    }

    fn iter(&self) -> TensorDataIter {
        match self {
            TensorData::Scalar(_) => TensorDataIter::Scalar(std::iter::once(self)),
            TensorData::NTensor(vec) => TensorDataIter::NTensor(vec.iter()),
        }
    }

    fn shape_as_string(&self) -> String {
       self.shape().iter().map(|x| x.to_string()).collect::<Vec<_>>().join("x") 
    }

    fn filled(value: f32, shape: Vec<usize>) -> TensorData {
        if shape.is_empty() {
            return TensorData::Scalar(value);
        }

        TensorData::NTensor(vec![TensorData::filled(value, shape[1..].to_vec()); 
                                 shape[0]])
    }

    fn perform_operand<F>(&self, other: &TensorData, operand: F) -> TensorData  // Used for add,
                                                                                // times, etc.
    where
        F: Fn(f32, f32) -> f32, 
    {
        if self.shape() != other.shape() {
            panic!("Mismatched shapes: S ({}) - O ({})",
                   self.shape_as_string(), other.shape_as_string());
        }

        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(operand(*a, *b)),
            (TensorData::NTensor(ten_a), TensorData::NTensor(ten_b)) => 
                if self.shape() != other.shape() {
                    panic!("Attemped operand on tensors of different shapes:\n\t[{}]\n\t[{}]",
                           self.shape_as_string(),
                           other.shape_as_string())
                } else {
                    TensorData::NTensor(ten_a.iter()
                                         .zip(ten_b)
                                         .map(|(a, b)| a.perform_operand(b, &operand))
                                         .collect::<Vec<_>>())
                }
            (TensorData::NTensor(ten), scalar) | (scalar, TensorData::NTensor(ten)) => 
                TensorData::NTensor(ten.iter()
                                     .map(|t| t.perform_operand(scalar, &operand))
                                     .collect::<Vec<_>>())
        }
    }

    fn len(&self, dim: usize) -> usize {
        match self {
            TensorData::Scalar(_) => if dim == 0 {1} else {panic!("Scalars don't have dimensions")},
            TensorData::NTensor(ten) => {
                if dim >= self.rank() {
                    panic!("Dimension {} is out of bounds for tensor of rank {}", dim, self.rank());
                }
                if dim == 0 {
                    ten.len()
                } else {
                    // Assume all sub-tensors have the same shape
                    ten[0].len(dim - 1)
                }
            }
        }
    }

    pub fn rank(&self) -> usize {
        match self {
            TensorData::Scalar(_) => 0,
            TensorData::NTensor(vec) => vec[0].rank() + 1,
        }
    }

    fn tensor_product(&self, other: &TensorData) -> TensorData {
        match self {
            TensorData::Scalar(val) => other * *val,
            TensorData::NTensor(ten) => TensorData::NTensor(ten.iter()
                                                                     .map(|t| t.tensor_product(other))
                                                                     .collect())
        }
    }

     fn contract(&self, ranks_to_contract: Vec<usize>) -> TensorData {
        match self {
            TensorData::Scalar(_) => panic!("Attempted to contract a scalar."),
            TensorData::NTensor(_) => {
                if ranks_to_contract.is_empty() {
                    panic!("Contraction requires at least 1 dimension, 0 given");
                }
                if ranks_to_contract.iter().any(|&r| r >= self.rank()) {
                    panic!("Given ranks {:?} contain a rank higher than that of the tensor ({})", ranks_to_contract, self.rank());
                }
                if ranks_to_contract.len() != ranks_to_contract.iter().collect::<std::collections::HashSet<_>>().len() {
                    panic!("Duplicate ranks in contraction list");
                }

                // Reverse the ranks to match the new logic
                let reversed_ranks: Vec<usize> = ranks_to_contract.iter().map(|&r| self.rank() - 1 - r).collect();
                let mut sorted_ranks = reversed_ranks;
                sorted_ranks.sort_unstable();

                // Check if dimensions to be contracted have the same length
                let dim_sizes: Vec<usize> = sorted_ranks.iter().map(|&r| self.len(r)).collect();
                if dim_sizes.windows(2).any(|w| w[0] != w[1]) {
                    panic!("Dimensions to be contracted must have the same length. Sizes: {:?}", dim_sizes);
                }

                // Helper function to perform contraction
                fn contract_helper(tensor: &TensorData, ranks: &[usize]) -> TensorData {
                    match tensor {
                        TensorData::Scalar(_) => panic!("Unexpected scalar encountered during contraction"),
                        TensorData::NTensor(data) => {
                            if tensor.rank() < ranks.len() {
                                panic!("Cannot contract tensor with rank less than number of contraction dimensions");
                            }
                            if ranks.is_empty() {
                                return tensor.clone();
                            }
                            if ranks[0] == 0 {
                                // Base case: contract the first dimension
                                let result = data.iter().fold(TensorData::Scalar(0.0), |acc, t| &acc + t);
                                if ranks.len() == 1 {
                                    // If no more dimensions to contract, return the result
                                    result
                                } else {
                                    // Continue contracting remaining dimensions
                                    contract_helper(&result, &ranks[1..])
                                }
                            } else {
                                // Recursive case: move towards the dimensions to contract
                                TensorData::NTensor(
                                    data.iter()
                                        .map(|t| contract_helper(t, &ranks.iter().map(|&r| r - 1).collect::<Vec<_>>()))
                                        .collect())
                            }
                        }
                    }
                }

                contract_helper(self, &sorted_ranks)
            }
        }
    }

    fn identity(size: usize, dimensions: usize) -> TensorData {
        if size <= 0 {
            panic!("Size must be > 0");
        }

        if dimensions == 0 {
            return TensorData::Scalar(1.0);
        }


        fn recursive_traverse(td: &TensorData, i: usize) -> TensorData {
            match td.clone() {
                TensorData::NTensor(mut vec) => {
                    match vec[i] {
                        TensorData::Scalar(_) => {
                            vec[i] = TensorData::Scalar(1.0);
                        },
                        TensorData::NTensor(_) => {
                            vec[i] = recursive_traverse(&vec[i], i);
                        },
                    }
                    
                    TensorData::NTensor(vec)
                },
                _ => panic!("Recursive traverse error, write this later!"),
            }
        }
        
        TensorData::filled(0.0, vec![size; dimensions]).iter()
                                                       .zip(0..size)
                                                       .map(|(t, i)| {
                                                            let mut t_clone = t.clone();
                                                            t_clone[i] = recursive_traverse(&t_clone[i], i);
                                                            t_clone
                                                       })
                                                       .collect()
    }

    fn get_element(&self, dim: usize, index: usize) -> TensorData {
        match self {
            TensorData::Scalar(_) => self.clone(),
            TensorData::NTensor(ten) => {
                if dim == 0 {
                    ten[index].clone()
                } else {
                    ten[0].get_element(dim - 1, index)
                }
            }
        }
    }

    fn get_slice(&self, dim: usize, index: usize) -> TensorData {
        match self {
            TensorData::Scalar(_) => self.clone(),
            TensorData::NTensor(ten) => {
                if dim == 0 {
                    ten[index].clone()
                } else {
                    TensorData::NTensor(
                        ten.iter().map(|t| t.get_slice(dim - 1, index)).collect()
                    )
                }
            }
        }
    }

   fn transpose(&self, mut a: usize, mut b: usize) -> TensorData {
        if a == b {
            return self.clone();
        }
       
        if b < a {  // Ensure a is always the smaller index
            std::mem::swap(&mut a, &mut b);
        }
        
        match self {
            TensorData::Scalar(v) => TensorData::Scalar(*v),
            TensorData::NTensor(ten) => {
                if a == 0 {
                    if b == self.rank() - 1 {  // Swap with innermost
                        let mut new = Vec::new();
                        for i in 0..ten[0].len(b - 1) {
                            new.push(TensorData::NTensor(
                                ten.iter().map(|t| t.get_element(b - 1, i)).collect()
                            ));
                        }
                        TensorData::NTensor(new)
                    } else {  // Swap with other (not innermost)
                        let mut new = Vec::new();
                        for i in 0..ten[0].len(b - 1) {
                            new.push(TensorData::NTensor(
                                ten.iter().map(|t| t.get_slice(b - 1, i)).collect()
                            ));
                        }
                        TensorData::NTensor(new)
                    }
                } else {  // Swap is 'deeper'
                    TensorData::NTensor(
                        ten.iter()
                           .map(|t| t.transpose(a - 1, b - 1))
                           .collect()
                    )
                }
            }
        }
    }

    fn full_transpose(&self, new_order: Vec<usize>) -> TensorData {
        match self {
            TensorData::Scalar(_) => self.clone(),
            TensorData::NTensor(_) => if new_order.len() != self.rank() {
                panic!("Transpose attempt had wrong amount of columns specified ({}), need {}", 
                    new_order.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
                    &self.rank())
            } else if new_order.iter().any(|x| x >= &self.rank()) {
                panic!("Transpose attempt had invalid index ({}), max is {} (0-based indexing)",
                    new_order.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "),
                    self.rank()-1)
            } else {
                let swaps = get_least_swaps(&mut (0..self.rank() as i32).map(|x| x as usize).collect(), &mut new_order.clone());
                swaps.iter().fold(self.clone(), |new, (a, b)| new.transpose(*a, *b))
            }
        }
    }

    fn dot_product(&self, other: &TensorData) -> TensorData {
        match (self, other) {
            (TensorData::Scalar(a), TensorData::Scalar(b)) => TensorData::Scalar(a * b),
            (TensorData::NTensor(ten), TensorData::Scalar(_)) | (TensorData::Scalar(_), TensorData::NTensor(ten)) => 
                TensorData::NTensor(ten.iter()
                                   .map(|t| t.dot_product(other))
                                   .collect()),
            (TensorData::NTensor(_), TensorData::NTensor(_)) => 
                self.tensor_product(other)  // Contracting innermost dim of LHS and outermost of RHS
                    .contract(vec![self.rank()-1, self.rank()])
        }
    }

    fn hamarand_product(&self, other: &TensorData) -> TensorData {
        // Neither scalar and unmatching.
        if self.shape() != other.shape() && (self.rank() > 0 && other.rank() > 0) {
            panic!("Mismatched dimensions: S ({}) - O ({})",
                   self.shape_as_string(), other.shape_as_string());
        }

        TensorData::perform_operand(self, other, Mul::mul)
    }

    fn value(&self) -> f32 {
        match self {
            TensorData::Scalar(val) => *val,
            TensorData::NTensor(_) => panic!("Attempted to retrieve the value of a non-scalar.")
        }
    }
}



#[derive(Clone, Debug)]
pub struct Tensor<const RANK: usize> {
    data: TensorData,
    _phantom: PhantomData<[(); RANK]>,
}

// Implementations for Tensor<RANK>

impl<const RANK: usize> Index<usize> for Tensor<RANK> {
    type Output = Tensor<{RANK - 1}>;

    fn index(&self, index: usize) -> &Self::Output {
        match &self.data {
            TensorData::Scalar(_) => panic!("Attempted to index a scalar value"),
            TensorData::NTensor(vec) => {
                &Tensor {
                    data: vec[index].clone(),
                    _phantom: PhantomData,
                }
            }
        }
    }
}

impl<const RANK: usize> IndexMut<usize> for Tensor<RANK> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match &mut self.data {
            TensorData::Scalar(_) => panic!("Attempted to index a scalar value"),
            TensorData::NTensor(vec) => {
                &mut Tensor {
                    data: vec[index].clone(),
                    _phantom: PhantomData,
                }
            }
        }
    }
}

impl<T, const RANK: usize> FromIterator<T> for Tensor<RANK>
where
    T: Clone,
    [(); RANK + 1]: Sized,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let vec: Vec<T> = iter.into_iter().collect();

        Tensor {
            data: TensorData::NTensor(vec),
            _phantom: PhantomData,
        }
    }
}

enum TensorIter<'a, const RANK: usize> {
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
    pub fn from_td(td: &TensorData) -> Tensor<RANK> {
        Tensor {
            data: td.clone(),
            _phantom: PhantomData
        }
    }

    fn iter(&self) -> TensorIter<RANK> {
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
        F: Fn(f32, f32) -> f32, 
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

    // TODO: Proper tensor/dot product and contract exposure (operations with tensors of different
    // ranks).
}


// All operators with 2 tensors are Hamarand style
// (except for multiplication, which is the dot product)

macro_rules! impl_tensor_data_op {
    ($trait:ident, $method:ident, $op:expr) => {
        impl $trait for &TensorData {
            type Output = TensorData;
            fn $method(self, rhs: Self) -> Self::Output {
                self.perform_operand(rhs, $op)
            }
        }

        impl $trait<f32> for &TensorData {
            type Output = TensorData;
            fn $method(self, rhs: f32) -> Self::Output {
                self.perform_operand(&TensorData::Scalar(rhs), $op)
            }
        }

        impl $trait<f32> for TensorData {
            type Output = TensorData;
            fn $method(self, rhs: f32) -> Self::Output {
                self.perform_operand(&TensorData::Scalar(rhs), $op)
            }
        }

        impl $trait<&TensorData> for f32 {
            type Output = TensorData;
            fn $method(self, rhs: &TensorData) -> Self::Output {
                rhs.perform_operand(&TensorData::Scalar(self), |a, b| $op(b, a))
            }
        }

        impl $trait<TensorData> for f32 {
            type Output = TensorData;
            fn $method(self, rhs: TensorData) -> Self::Output {
                (&rhs).perform_operand(&TensorData::Scalar(self), |a, b| $op(b, a))
            }
        }
    }
}

impl_tensor_data_op!(Add, add, |a, b| a + b);
impl_tensor_data_op!(Sub, sub, |a, b| a - b);
impl_tensor_data_op!(Div, div, |a, b| a / b);

// Special implementation for Mul to handle dot product for TensorData
impl Mul for &TensorData {
    type Output = TensorData;
    fn mul(self, rhs: Self) -> Self::Output {
        self.dot_product(rhs)
    }
}

impl Mul<f32> for &TensorData {
    type Output = TensorData;
    fn mul(self, rhs: f32) -> Self::Output {
        self.perform_operand(&TensorData::Scalar(rhs), |a, b| a * b)
    }
}

impl Mul<f32> for TensorData {
    type Output = TensorData;
    fn mul(self, rhs: f32) -> Self::Output {
        self.perform_operand(&TensorData::Scalar(rhs), |a, b| a * b)
    }
}

impl Mul<&TensorData> for f32 {
    type Output = TensorData;
    fn mul(self, rhs: &TensorData) -> Self::Output {
        rhs.perform_operand(&TensorData::Scalar(self), |a, b| a * b)
    }
}

impl Mul<TensorData> for f32 {
    type Output = TensorData;
    fn mul(self, rhs: TensorData) -> Self::Output {
        (&rhs).perform_operand(&TensorData::Scalar(self), |a, b| a * b)
    }
}

// Keep the existing Tensor implementations
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


fn get_least_swaps(old: &mut Vec<usize>, new: &mut Vec<usize>) -> Vec<(usize, usize)> {
    let n = old.len();
    let mut new_map: HashMap<usize, usize> = HashMap::new();
    let mut visited = vec![false; n];
    let mut swaps = Vec::new();

    // Create a map of element to index for the new vector
    for (i, &val) in new.iter().enumerate() {
        new_map.insert(val, i);
    }

    for i in 0..n {
        if visited[i] || old[i] == new[i] {
            continue;
        }

        // Start of a new cycle
        let mut j = i;
        while !visited[j] {
            visited[j] = true;
            let next = new_map[&old[j]];
            if next != j {
                swaps.push((j, next));
                old.swap(j, next);
            }
            j = next;
        }
    }

    swaps
}


// TODO
// Flatten(to dim_n)
// Normalize
    // sum(all): x^2 = 1
// Eigen
// Cross product (Vector)

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

    pub fn cross_product(&self, other: &Vector) -> Scalar {
        if self.len(0) != 3 || other.len(0) != 3 {
            panic!("Tried to get the cross product of vectors of wrong length ({}, {}), expected 3.", self.len(0), other.len(0))
        }

        let mut scalar = Scalar::from(0.0);

        for (a, b) in self.data.iter().zip(other.data.iter()).collect::<Vec<(TensorData, TensorData)>>() {
            match (a, b) {
                (TensorData::Scalar(a), TensorData::Scalar(b)) => scalar = scalar + (a * b),
                _ => panic!("Encountered a tensor of rank > 1 while calculating cross product.")
            }
        }

        scalar
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

     fn get_cofactor(&self, p: usize, q: usize) -> f32 {
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
        let minor = Matrix::from(cofactor).unwrap();
        f32::powi(-1.0, (p + q) as i32) * minor.calc_determinant()
    }

    fn calc_determinant(&self) -> f32 {  // TODO: Rewrite without recursion + matrix creation and
                                          // compare speed.
        if self.shape()[0] == 2 {
            return self.data[0][0].value() * self.data[1][1].value() - self.data[0][1].value() * self.data[1][0].value();
        }

        (0..self.shape()[1]).map(|j| (self.get_cofactor(0, j) * self.data[0][j].clone()).value())
                            .sum()
    }

    fn get_cofactor_matrix(&self) -> Matrix {  // TODO: All this shit as attributes.
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

    fn get_determinant(&self) -> Result<f32, String> {
        if self.shape()[0] != self.shape()[1] {
            return Err(format!("Only square matrices have determinants ({}x{})", self.shape()[0], self.shape()[1]));
        }

        match self.shape()[0] {
            0 => Ok(1.0),
            1 => Ok(self.data[0][0].value()),
            _ => Ok(self.calc_determinant())
        }
    }

    fn inverse(&self) -> Result<Matrix, String> {
        let det: f32;

        match self.get_determinant() {
            Ok(d) => if d == 0.0 {return Err("Det == 0, so no inverse matrix exists.".to_string())} else {det = d},
            Err(e) => return Err(e)
        }
        
        let adjugate = self.get_cofactor_matrix().transpose(0, 1);

        Ok(&adjugate / det)
    }

    fn print(&self, title: &str) {
        println!("-------- {} ({}x{}) --------", title, self.shape()[0], self.shape()[1]);

        for row in self.data.iter() {
            println!("{:?}", row);
        }

        println!("-----------------------------");
    }
}

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

    (&matrix1 + &matrix1). print("Double");
    (&matrix1 - 3.0).print("Plus 3");

    matrix1.transpose(0, 1).print("Transposed");

    Matrix::identity(3, 2).transpose(0, 1).print("Identity");
    Matrix::filled(8.0, vec![3, 6]).transpose(0, 1).print("Filled");

    (&matrix1 * &matrix1.transpose(0, 1)).print("Times");

    println!("DETERMINANT: {}", matrix2.get_determinant().expect(""));
    matrix2.inverse().expect("").print("Inverse");
}
