CC = g++
CFLAGS = -Wall -Wclobbered -Wempty-body -Wignored-qualifiers -Wimplicit-fallthrough=3\
         -Wmissing-field-initializers -Wtype-limits -Wuninitialized -Wshift-negative-value\
         -Wunused-but-set-parameter\
         -std=c++0x -O2


INCLUDES = -I..

SRCS = main.cpp
.SUFFIXES:  .cpp  .o
OBJDIR = objs
OBJS = $(addprefix $(OBJDIR)/, $(SRCS:.cpp=.o))

MAIN = test

.PHONY: depend clean

all: $(OBJDIR) $(MAIN)
	@echo  $(MAIN) compiled

$(MAIN): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

$(OBJDIR)/%.o: %.cpp Lapack_dyn.hpp
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

$(OBJDIR):
	mkdir $(OBJDIR)

Lapack_dyn.hpp: ../Lapack.hpp
	sed -e 's/static //g' -e 's/int ilaenv/virtual int ilaenv/' -e 's/class Lapack/class Lapack_dyn/' -e 's/constexpr/const/' -e 's/void xerbla/virtual void xerbla/' -e 's/public:/public:\n\n    virtual ~Lapack_dyn(){}\n/' -e 's/\/\*const\*\//const/' $< > $@

clean:
	$(RM) $(OBJDIR)/*.o *~ $(MAIN) Lapack_dyn.hpp
	rmdir $(OBJDIR)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it
