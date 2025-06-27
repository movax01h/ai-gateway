import pytest

from ai_gateway.code_suggestions.processing.ops import LanguageId
from ai_gateway.code_suggestions.prompts.parsers import CodeParser

# editorconfig-checker-disable
PYTHON_SOURCE_SAMPLE = """
import os
import time

# more code
# more code
# more code
# more code
# more code
# more code

import random
import pandas as pd

def sum(a, b):
    import numpy as np
    return a + b

def subtract(a, b):
    return a - b

class Calculator:
    def __init__(self):
        self.result = 0

    def calculateSum(self, a, b):
        self.result = sum(a, b)
"""

C_SOURCE_SAMPLE = """
#include <stdio.h>
#include <stdlib.h>

// Define a struct called 'Person'
struct Person {
    char name[50];
    int age;
};

// Function to initialize a Person struct
void initializePerson(struct Person *person, const char *name, int age) {
    strcpy(person->name, name);
    person->age = age;
}

// Function to print the details of a Person
void printPersonDetails(const struct Person *person) {
    printf("Name: %s\n", person->name);
    printf("Age: %d\n", person->age);
}

int main() {
    struct Person p;
    initializePerson(&p, "John Doe", 25);
    printPersonDetails(&p);
    return 0;
}
"""

JAVASCRIPT_SOURCE_SAMPLE = """
import React, { useState } from "react";
import dateFns from "date-fns";
import { sum } from "mathjs";

const App = () => {
  const [date, setDate] = useState(new Date());
  const [number, setNumber] = useState(0);

  const addNumber = () => {
    setNumber(sum(number, 1));
  };

  const getDateString = () => {
    return dateFns.format(date, "YYYY-MM-DD");
  };

  return (
    <div>
      <h1>Date: {getDateString()}</h1>
      <h1>Number: {number}</h1>
      <button onClick={addNumber}>Add 1</button>
    </div>
  );
};

export default App;
"""

JAVASCRIPT_SAMPLE_SOURCE_2 = """
// Function Declaration 1
function add(a, b) {
  return a + b;
}

// Function Declaration 2
function multiply(a, b) {
  return a * b;
}

// Class Declaration
class Calculator {
  constructor() {
    this.result = 0;
  }

  // Method 1
  calculateSum(a, b) {
    this.result = add(a, b);
  }

  // Method 2
  calculateProduct(a, b) {
    this.result = multiply(a, b);
  }

  // Method 3
  getResult() {
    return this.result;
  }
}

// Generator function 1
function* countNumbers() {
  let i = 1;
  while (true) {
    yield i;
    i++;
  }
}

// Usage
const calculator = new Calculator();

calculator.calculateSum(5, 3);
console.log("Sum:", calculator.getResult()); // an inline comment

calculator.calculateProduct(5, 3);
console.log("Product:", calculator.getResult()); /* and a block comment */
"""

TS_SAMPLE_SOURCE = """
// Importing required modules
import { Calculator } from './calculator';

// Function 1: Add two numbers
function addNumbers(a: number, b: number): number {
  return a + b;
}

// Function 2: Multiply two numbers
function multiplyNumbers(a: number, b: number): number {
  return a * b;
}

// Main class
class MyApp {
  // Function to perform some calculations
  performCalculations(): void {
    const num1 = 5;
    const num2 = 3;

    const sum = addNumbers(num1, num2);
    console.log('Sum:', sum);

    const product = multiplyNumbers(num1, num2);
    console.log('Product:', product);
  }
}

// Instantiating the class and running the app
const myApp = new MyApp();
myApp.performCalculations();

"""

CPP_SAMPLE_SOURCE = """
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// comment 1
int main() {
  // Create a vector of strings.
  vector<string> strings = {"Hello", "World"};

  // Print the vector of strings.
  for (string string : strings) {
    cout << string << endl;
  }

  return 0;
}
"""

CSHARP_SAMPLE_SOURCE = """
using System.Console;

// comment 1
public class Program
{
  public static void Main()
  {
      Console.WriteLine("Hello, world!");
  }
}
"""

GO_SAMPLE_SOURCE = """
package main

// comment 1
import (
    "fmt"
    "log"
    "net/http"
    "os"
)

// comment 2
func main() {
    http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
        fmt.Fprintf(w, "Hello, world!")
    })

    log.Fatal(http.ListenAndServe(":8080", nil))
}
"""

GO_SAMPLE_SOURCE_2 = """
package main

import "fmt"
import "log"
"""

JAVA_SAMPLE_SOURCE = """
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

// comment 1
@SpringBootApplication
public class Application {

    public static void main(String[] args) {
        SpringApplication.run(Application.class, args);
    }
}

/* block comment 1 */
"""

RUBY_SAMPLE_SOURCE = """
require 'date'
require_relative 'lib/test'

# comment 1
module Application
  class Test
    def hello(self, test)
      puts "hello world"
    end
  end
end
"""

RUST_SAMPLE_SOURCE = """
use std::io;
use actix_web::{web, App, HttpServer};

// comment 1
fn main() {
    let server = HttpServer::new(|| {
        App::new()
            .service(web::get("/hello").to(hello))
    });

    server.bind("127.0.0.1:8080").unwrap().run().unwrap();
}

/* block comment 1 */
fn hello() -> impl Responder {
    "Hello, world!"
}
"""

SCALA_SAMPLE_SOURCE = """
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

class Person

def greet(name: String): Unit = {
  println(s"Hello, $name!")
}

// comment 1
object Main extends App {

  val numbers = ArrayBuffer[Int]()
  for (i <- 1 to 10) {
    numbers += Random.nextInt(100)
  }

  println(numbers.sorted)
}
"""

PHP_SAMPLE_SOURCE = """
<?php

use SomeOtherNamespaceCoolFunction;

// Import 1
require_once 'calculator.php';

// Function 1: Add two numbers
function addNumbers($a, $b) {
    return $a + $b;
}

// Function 2: Multiply two numbers
function multiplyNumbers($a, $b) {
    return $a * $b;
}

// Main class
class MyApp {
    // Function to perform some calculations
    public function performCalculations() {
        $num1 = 5;
        $num2 = 3;

        $sum = addNumbers($num1, $num2);
        echo 'Sum: ' . $sum . PHP_EOL;

        $product = multiplyNumbers($num1, $num2);
        echo 'Product: ' . $product . PHP_EOL;
    }
}

// Instantiating the class and running the app
$myApp = new MyApp();
$myApp->performCalculations();
?>
"""

KOTLIN_SAMPLE_SOURCE = """
import kotlin.collections.*
import java.util.Random

// Define a dice
class Dice(val sides: Int) {
    private val random = Random()

    fun roll(): Int {
        return random.nextInt(sides) + 1
    }
}

/* This is a main function */
fun main() {
    val sixSidedDice = Dice(6)
    val rollResult = sixSidedDice.roll()
    println("Rolled a $rollResult")
}
"""
# editorconfig-checker-enable


@pytest.mark.parametrize(
    ("lang_id", "source_code", "target_symbols_counts"),
    [
        (
            LanguageId.PYTHON,
            PYTHON_SOURCE_SAMPLE,
            {
                "import_statement": 5,
                "function_definition": 4,
                "comment": 6,
                "class_definition": 1,
            },
        ),
        (
            LanguageId.C,
            C_SOURCE_SAMPLE,
            {"preproc_include": 2, "function_definition": 3, "comment": 3},
        ),
        (LanguageId.JS, JAVASCRIPT_SOURCE_SAMPLE, {"import_statement": 3}),
        (
            LanguageId.TS,
            TS_SAMPLE_SOURCE,
            {
                "import_statement": 1,
                "function_declaration": 2,
                "comment": 6,
                "class_declaration": 1,
            },
        ),
        (
            LanguageId.CPP,
            CPP_SAMPLE_SOURCE,
            {"preproc_include": 3, "function_definition": 1, "comment": 3},
        ),
        (
            LanguageId.CSHARP,
            CSHARP_SAMPLE_SOURCE,
            {"using_directive": 1, "class_declaration": 1, "comment": 1},
        ),
        (
            LanguageId.GO,
            GO_SAMPLE_SOURCE,
            {"import_declaration": 1, "function_declaration": 1, "comment": 2},
        ),
        (LanguageId.GO, GO_SAMPLE_SOURCE_2, {"import_declaration": 2}),
        (
            LanguageId.JAVA,
            JAVA_SAMPLE_SOURCE,
            {
                "import_declaration": 2,
                "line_comment": 1,
                "class_declaration": 1,
                "block_comment": 1,
            },
        ),
        (
            LanguageId.RUBY,
            RUBY_SAMPLE_SOURCE,
            {"require": 2, "comment": 1, "module": 1, "class": 1},
        ),
        (
            LanguageId.RUST,
            RUST_SAMPLE_SOURCE,
            {
                "use_declaration": 2,
                "line_comment": 1,
                "block_comment": 1,
                "function_item": 2,
            },
        ),
        (
            LanguageId.SCALA,
            SCALA_SAMPLE_SOURCE,
            {
                "import_declaration": 2,
                "comment": 1,
                "class_definition": 1,
                "function_definition": 1,
            },
        ),
        (
            LanguageId.JS,
            JAVASCRIPT_SAMPLE_SOURCE_2,
            {
                "class_declaration": 1,
                "function_declaration": 2,
                "generator_function_declaration": 1,
                "comment": 10,
            },
        ),
        (
            LanguageId.PHP,
            PHP_SAMPLE_SOURCE,
            {
                "namespace_use_declaration": 1,
                "function_definition": 2,
                "comment": 6,
                "class_declaration": 1,
            },
        ),
        (
            LanguageId.KOTLIN,
            KOTLIN_SAMPLE_SOURCE,
            {
                "import_header": 2,
                "line_comment": 1,
                "class_declaration": 1,
                "function_declaration": 2,
                "multiline_comment": 1,
            },
        ),
    ],
)
@pytest.mark.asyncio
async def test_symbol_counter(
    lang_id: LanguageId,
    source_code: str,
    target_symbols_counts: dict[str, int],
):
    parser = await CodeParser.from_language_id(source_code, lang_id)
    output = parser.count_symbols()

    assert len(output) == len(target_symbols_counts)
    for symbol, expected_count in target_symbols_counts.items():
        assert output[symbol] == expected_count
