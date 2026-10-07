#include "Vlfsr16_free.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <ctime>
int main(int argc, char** argv){
  long B = atol(argv[1]), C = atol(argv[2]);
  Vlfsr16_free* top = new Vlfsr16_free;
  uint64_t acc = 0;
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
  for(long b=0;b<B;b++){
    top->clk=0; top->state=(uint16_t)((b*2654435761u)|1u); top->eval();
    for(long c=0;c<C;c++){ top->clk=1; top->eval(); top->clk=0; top->eval(); }
    acc += top->state;
  }
  clock_gettime(CLOCK_MONOTONIC,&t1);
  double s=(t1.tv_sec-t0.tv_sec)+(t1.tv_nsec-t0.tv_nsec)*1e-9;
  fprintf(stderr,"lfsr16_free acc=%lu  cyc*batch/s=%.3e  time=%.3fs\n",(unsigned long)acc,(double)B*C/s,s);
  delete top; return 0;
}
