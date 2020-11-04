//
// Created by neville on 03.11.20.
//

#ifndef SHAONGPU_SHA256_ON_CPU_H
#define SHAONGPU_SHA256_ON_CPU_H

#include <assert.h>
#include <string>
#include "padding.h"
#include "main_loop_cpu.h"

// return vector is currently passed by value, could be optimized
std::string sha256_on_cpu(const std::string in) {

    // 1. Padding
    std::vector<int> padded = padding(in);

    // 2. Nain Loop cpu
    std::vector<int> res_int = main_loop_cpu(padded);

    // 3. Convert Result to String
    std::string res_string = "";
    char buffer[50];
    for (int i = 0; i < res_int.size(); i++) {
        int curr = __builtin_bswap32(res_int[i]);
        sprintf(buffer, "%x", curr);
        res_string += buffer;

    }
    return res_string;

}

void sha256_on_cpu_test() {

    std::string out = sha256_on_cpu("");
    assert(out == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");

    out = sha256_on_cpu("Neville");
    assert(out == "359ef5c170178ca7309c92222e3a03707f03194eed16b262ab35536cbd72536f");

    out = sha256_on_cpu("Neville");
    assert(out == "a3ef49d473eec07b75d8a7a93a71b1f46b0b7573aa35e4789b7572c52acff793");

    out = sha256_on_cpu(
            "Der Begriff Secure Hash Algorithm (kurz SHA, englisch für sicherer Hash-Algorithmus) bezeichnet eine Gruppe standardisierter kryptologischer Hashfunktionen. Diese dienen zur Berechnung eines Prüfwerts für beliebige digitale Daten (Nachrichten) und sind unter anderem die Grundlage zur Erstellung einer digitalen Signatur.\n"
            "Der Prüfwert wird verwendet, um die Integrität einer Nachricht zu sichern. Wenn zwei Nachrichten den gleichen Prüfwert ergeben, soll die Gleichheit der Nachrichten nach normalem Ermessen garantiert sein, unbeschadet gezielter Manipulationsversuche an den Nachrichten. Darum fordert man von einer kryptologischen Hashfunktion die Eigenschaft der Kollisionssicherheit: es soll praktisch unmöglich sein, zwei verschiedene Nachrichten mit dem gleichen Prüfwert zu erzeugen. ");
    assert(out == "598ae7a5cd9e62a4b605063f1f353a82aeb75a854a35e322c710275ed3f82883");

    out = sha256_on_cpu("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(out == "5df6e0e2761359d30a8275058e299fcc0381534545f55cf43e41983f5d4c9456");


}


#endif //SHAONGPU_SHA256_ON_GPU_H
