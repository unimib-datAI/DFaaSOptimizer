// Alternate half of level-2 candidates with one-step eviction and admission.
if (level==2 && c%2==0 && eligible_shape[0]>0) {
  e=eligible[(rng+c)%eligible_shape[0]];
  i=src[e]; j=dst[e]; f=fun[e];
  long want=z[row+i*nf+f], old_space=space[row+j*nf+f];
  if (want>old_space && inc[row+i*nf+f]==0 && z[row+j*nf+f]==load[j*nf+f]-x[j*nf+f]) {
    long begin=in_starts[j], size=in_starts[j+1]-begin;
    rng=(rng*16807)%2147483647L;
    long offset=rng%size, num=0, donors[2]={-1,-1}, remove[2]={0,0};
    for(long n=0;n<size && num<2;n++) {
      long d=in_edges[begin+(offset+n)%size], g=fun[d], amount=y[p*ne+d];
      if(g==f || amount<=0) continue;
      long needed=release[row+j*nf+g];
      if(num && fun[donors[0]]==g) needed-=remove[0];
      long q=c%8==0 ? amount : min(amount,max(0L,needed));
      if(q<=0) continue;
      donors[num]=d; remove[num]=q; num++;
      if(c%4==2) break;
    }
    if(num) {
      long used=0;
      for(long g=0;g<nf;g++) if(g!=f) {
        long work=x[j*nf+g]+inc[row+j*nf+g];
        for(long k=0;k<num;k++) if(fun[donors[k]]==g) work-=remove[k];
        used+=costs[(j*nf+g)*stride+work];
      }
      long copies=(ram[j]-used)/memory[f];
      long work=x[j*nf+f]+inc[row+j*nf+f];
      long admit=min(want,max(0L,limits[(j*nf+f)*params[6]+copies]-work));
      if(admit>old_space) {
        edges[0]=e; delta[0]=admit;
        for(long k=0;k<num;k++) {edges[k+1]=donors[k];delta[k+1]=-remove[k];}
      }
    }
  }
} else
