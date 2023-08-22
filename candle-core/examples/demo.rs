#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::Result;
use candle_core::{Device, Tensor};
use std::{collections::HashMap, rc::Rc};
struct Cacher<T>
where
    T: Fn(u32) -> u32,
{
    query: T,
    value: Option<u32>,
}

fn fn_once<F>(mut func: F)
where
    F: FnMut(usize) -> bool,
{
    println!("{}", func(1));
}

fn main() {
    let mut x = vec![1, 2, 3];
    fn_once(|z| {
        x[z] = 10;
        x.len() == 3;
        true
    });
    let names = ["sunface".to_string(), "sunfei".to_string()];
    let ages = [18, 18];
    let folks: HashMap<_, _> = names.iter().zip(ages.into_iter()).collect();
    struct A(i32);
    let cc = &A(1);
    let t = cc.clone();

    let v = &mut String::from("value");
    let c = Rc::new(v);
    use std::cell::RefCell;
    let s = Rc::new(RefCell::new("我很善变，还拥有多个主人".to_string()));
    let mut s = s.borrow_mut();
    s.push_str("string");

    println!("{:?}", x);
}
