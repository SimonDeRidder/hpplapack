#include <iostream>
#include <sstream>
#include <fstream>

#include "Testing_eig.h"
#include "Testing_lin.h"

using namespace std;

int main(int argc, char* argv[])
{
    Testing_eig<double>* te = new Testing_eig<double>();
    // test dgebal
    ifstream f("data/dbal.in");
    char dummy[20];
    f.getline(dummy, 20);
    te->dchkbl(f,cout);
    f.close();
    delete te;
    // TODO: add all other tests
}