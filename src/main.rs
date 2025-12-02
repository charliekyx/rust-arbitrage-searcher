use std::collections::HashMap;

const GLOAL_VAR: &str = "my name is ";
fn testF() {
    let vv: i32 = -786;
    let ab = String::from("Hello, world!");
    let ss = "hell";
    println!("hello again {}", vv);
    println!("amy");
    println!("amy");
    println!("{}", GLOAL_VAR);

    let mut actualstring = "abc".to_string();
    actualstring.push_str(" edf");

    let oo = 78;

    if vv > oo {
        println!("impossible")
    } else {
        println!("hahah")
    }

    let key = 10;
    match key {
        // switch case
        1 => {
            let c = key + oo;
        }
        10 => {
            // do nothing
        }
        _ => {} // default case
    }

    let mut cnt = 0;
    loop {
        if cnt > 10 {
            break;
        }
        cnt += 1;
    }

    while cnt < 20 {
        cnt += 1
    }

    // not including 100
    for i in 0..100 {
        println!("{}", i)
    }

    let mut capitalCities = HashMap::new();
    capitalCities.insert("France", "Paris");
    capitalCities.insert("Japan", "Tokyo");

    let numbers = [1, 2, 3, 4, 5];
    println!("{:?}", numbers);

    let fruits = ["apple", "banana", "orange"];
    for fruit in fruits {
        println!("I like {}.", fruit);
    }

    let mut fruits = vec!["apple", "banana"];
    fruits.push("cherry");

    println!("{:?}", fruits); // ["apple", "banana", "cherry"]
    fruits.insert(0, "apple"); // add to a specific index
    fruits.pop();

    // let mut scores = hashmap![
    //     "Alice" => 10,
    //     "Bob" => 20,
    //     "Charlie" => 30,
    // ];
}

fn main() {
    println!("hello");
    testF();
}
