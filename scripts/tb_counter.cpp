#include "Vcounter8.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
int main(int argc, char** argv){
  long B = atol(argv[1]), C = atol(argv[2]);
  Vcounter8* top = new Vcounter8;
  uint64_t acc = 0;
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  for(long b=0;b<B;b++){
    top->clk=0; top->rst=1; top->en=1; top->eval();
    top->clk=1; top->eval(); top->clk=0; top->rst=0; top->eval();  // reset once
    for(long c=0;c<C;c++){ top->clk=1; top->eval(); top->clk=0; top->eval(); }
    acc += top->cnt;
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
  fprintf(stderr,"counter8 acc=%lu  cyc*batch/s=%.3e  time=%.3fs\n",(unsigned long)acc,(double)B*C/s,s);
  delete top; return 0;
}
