#include <iostream>
#include <sstream>
#include <fstream>

#include "Testing_eig.hpp"
#include "Testing_lin.hpp"

using namespace std;

int main(int argc, char* argv[])
{
	// TEST EIG ROUTINES
	Testing_eig<double>* te = new Testing_eig<double>();
	// test dgebal
	// TODO integrate into dchkee
	ifstream f("data/dbal.in");
	char dummy[20];
	f.getline(dummy, 20);
	te->dchkbl(f,cout);
	f.close();
	// TODO: add all other tests
	delete te;
	// TEST LIN ROUTINES
	const int NMAX = 132;
	const int MAXIN = 12;
	const int MAXRHS = 16;
	const int MATMAX = 30;
	Testing_lin<double>* tl = new Testing_lin<double>();
	// test dgebal
	f.open("data/dtest.in");
	tl->dchkaa(NMAX, MAXIN, MAXRHS, MATMAX, f, cout);
	f.close();
	// TODO: add all other tests
	delete tl;
}