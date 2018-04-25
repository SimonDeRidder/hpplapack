CC = g++
CFLAGS = -Wall -g -std=c++0x -O2


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

$(OBJDIR)/%.o: %.cpp $(HDRS)
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

$(OBJDIR):
	mkdir $(OBJDIR)

Lapack_dyn.h: ../Lapack.h
	sed -e 's/static //' -e 's/int ilaenv/virtual int ilaenv/' $< > $@

clean:
	$(RM) $(OBJDIR)/*.o *~ $(MAIN) Lapack_dyn.h
	rmdir $(OBJDIR)

depend: $(SRCS)
	makedepend $(INCLUDES) $^

# DO NOT DELETE THIS LINE -- make depend needs it