# TODO

## Test Correctness

* ~~Write program that runs all tests~~
* ~~Create input strings in ASCII with corresponding SHA-256 to test whole SHA-256 program (Maybe even call another SHA-256 library internally to verify correctness of our kernel)~~
* Create tests for for all Methods
    * ~~Padding~~
    * ~~Choice~~
    * ~~Majority~~
    * ~~Sigma_0~~
    * Sigma_1
    * sigma_o
    * sigma_1
    * ROTL
    * ROTR
    * Message Schedule





## SHA-256
* ~~Main Method: char[] -> char[] (Neville)~~
* ~~Padding: int[] -> int[] (Neville)~~
    * ~~No conversion from char[] to int[] necessary, we can just interpret the chars as integers.~~
* ~~Prepare Message Schedule: int[] -> int (Neville)~~
    * ~~We cant compute everything in advance, since it would need to many registers. We need only the last 16 simultaniously.~~
* ~~Main Loop (Neville)~~
* ~~Choice Method: int,int,int -> int (Neville)~~
* ~~Majority Method: int,int,int -> int (Neville)~~
* ~~Sigma_0 Method: int -> int (Neville)~~
* Sigma_1 Method: int -> int (Basil)
* sigma_o Method: int -> int (Basil)
* sigma_1 Method: int -> int (Basil)
* ROTL Method: int,int -> int (Basil)
* ROTR Method: int,int -> int (Basil)
* Add constants + initial Hash values (Basil)




## PARSHA-256
* Finish first SHA-256
* Read paper 