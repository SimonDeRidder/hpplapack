#include <iostream>
#include <sstream>
#include <fstream>

#include "Testing_eig.h"
#include "Testing_lin.h"

using namespace std;

int main(int argc, char* argv[])
{
    // test dgebal
    ifstream f("data/dbal.in");
    char dummy[20];
    f.getline(dummy, 20);
    Testing_eig<double>::dchkbl(f,cout);
    f.close();
    // TODO: add all other tests
}