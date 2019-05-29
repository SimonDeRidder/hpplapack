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
	ifstream f;
	int const ENMAX   = 132;
	int const ENCMAX  = 20;
	int const ENEED   = 14;
	int const ELWORK  = ENMAX*(5*ENMAX+5)+1;
	int const ELIWORK = ENMAX*(5*ENMAX+20);
	int const EMAXIN  = 20;
	int const EMAXT   = 30;
	//// NEP
	//cout << "NEP: Testing Nonsymmetric Eigenvalue Problem routines" << endl;
	//f.open("data/nep.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// SEP
	//cout << "SEP: Testing Symmetric Eigenvalue Problem routines" << endl;
	//f.open("data/sep.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// SE2
	//cout << "SEP: Testing Symmetric Eigenvalue Problem routines" << endl;
	//f.open("data/se2.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	// SVD
	cout << "SVD: Testing Singular Value Decomposition routines" << endl;
	f.open("data/svd.in");
	te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	f.close();
	//// DEC
	//cout << "DEC: Testing DOUBLE PRECISION Eigen Condition Routines" << endl;
	//f.open("data/dec.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	// DEV
	cout << "DEV: Testing DOUBLE PRECISION Nonsymmetric Eigenvalue Driver" << endl;
	f.open("data/ded.in");
	te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	f.close();
	//// DGG
	//cout << "DGG: Testing DOUBLE PRECISION Nonsymmetric Generalized Eigenvalue Problem routines"
	//     << endl;
	//f.open("data/dgg.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// DGS, DGV, DGX & DXV
	//cout << "DGD: Testing DOUBLE PRECISION Nonsymmetric Generalized Eigenvalue Problem driver "
	//        "routines" << endl;
	//f.open("data/dgd.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// DSB
	//cout << "DSB: Testing DOUBLE PRECISION Symmetric Eigenvalue Problem routines" << endl;
	//f.open("data/dsb.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// DSG
	//cout << "DSG: Testing DOUBLE PRECISION Symmetric Generalized Eigenvalue Problem routines"
	//     << endl;
	//f.open("data/dsg.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	// DBL
	cout << "DGEBAL: Testing the balancing of a DOUBLE PRECISION general matrix" << endl;
	f.open("data/dbal.in");
	te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	f.close();
	//// DBK
	//cout << "DGEBAK: Testing the back transformation of a DOUBLE PRECISION balanced matrix"
	//     << endl;
	//f.open("data/dbak.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// DGL
	//cout << "DGGBAL: Testing the balancing of a pair of DOUBLE PRECISION general matrices" << endl;
	//f.open("data/dgbal.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// DGK
	//cout << "DGGBAK: Testing the back transformation of a pair of DOUBLE PRECISION balanced "
	//        "matrices" << endl;
	//f.open("data/dgbak.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// DBB
	//cout << "DBB: Testing banded Singular Value Decomposition routines" << endl;
	//f.open("data/dbb.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// GLM
	//cout << "GLM: Testing Generalized Linear Regression Model routines" << endl;
	//f.open("data/glm.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// GQR
	//cout << "GQR: Testing Generalized QR and RQ factorization routines" << endl;
	//f.open("data/gqr.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// GSV
	//cout << "GSV: Testing Generalized Singular Value Decomposition routines" << endl;
	//f.open("data/gsv.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	//// CSD
	//cout << "CSD: Testing CS Decomposition routines" << endl;
	//f.open("data/csd.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	////LSE
	//cout << "LSE: Testing Constrained Linear Least Squares routines" << endl;
	//f.open("data/lse.in");
	//te->dchkee(ENMAX, ENCMAX, ENEED, ELWORK, ELIWORK, EMAXIN, EMAXT, f, cout);
	//f.close();
	delete te;

	// TEST LIN ROUTINES
	int const LNMAX = 132;
	int const LMAXIN = 12;
	int const LMAXRHS = 16;
	int const LMATMAX = 30;
	Testing_lin<double>* tl = new Testing_lin<double>();
	// test dgebal
	f.open("data/dtest.in");
	tl->dchkaa(LNMAX, LMAXIN, LMAXRHS, LMATMAX, f, cout);
	f.close();
	// TODO: add all other tests
	delete tl;
}