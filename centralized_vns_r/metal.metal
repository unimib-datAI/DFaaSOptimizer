// All GPU arithmetic is integer; float64 policy is supplied by the host.
long idx=thread_position_in_grid.x;
long c=idx%params[2], p=idx/params[2], nf=params[4], nodes=params[3];
long ne=src_shape[0], stride=params[5], row=p*nodes*nf;
long edges[4]={-1,-1,-1,-1}, delta[4]={0,0,0,0};
if (ne>0) {
  long rng=(params[0]%2147483646L+1+104729*p+13007*params[1]+433*c)%2147483647L;
  rng=(rng*16807)%2147483647L;
  long e=(rng+c)%ne, i=src[e], j=dst[e], f=fun[e], level=levels[p];
  if (level==0) {
    long q=min(z[row+i*nf+f],space[row+j*nf+f]);
    if(c%3==0) q=min(q,1L);
    else if(c%3==1) q=min(q,fill[row+j*nf+f]);
    edges[0]=e; delta[0]=q;
  } else if (level==3) {
    long strength=1+(params[1]/9)%4;
    for(long a=0;a<strength;a++) {
      rng=(rng*16807)%2147483647L;
      long edge=rng%ne; bool duplicate=false;
      for(long b=0;b<a;b++) if(edges[b]==edge) duplicate=true;
      if(!duplicate) {edges[a]=edge; delta[a]=-y[p*ne+edge];}
    }
  } else if (level==1) {
    long begin=starts[i*nf+f], end=starts[i*nf+f+1];
    rng=(rng*16807)%2147483647L;
    long target=begin+rng%(end-begin), q=y[p*ne+e];
    if(c%3==0) q=min(q,1L);
    else if(c%3==1) q=min(q,release[row+j*nf+f]);
    edges[0]=e; delta[0]=-q; edges[1]=target; delta[1]=q;
  } else if(c%4>=2) {
    long begin=in_starts[j],end=in_starts[j+1];
    if(begin<end && z[row+i*nf+f]>0) {
      rng=(rng*16807)%2147483647L;
      long other=in_edges[begin+rng%(end-begin)],origin=src[other],g=fun[other];
      begin=starts[origin*nf+g];end=starts[origin*nf+g+1];
      rng=(rng*16807)%2147483647L;
      long target=begin+rng%(end-begin);
      if(dst[target]!=j && y[p*ne+other]>0) {
        long q1=y[p*ne+other];
        if(c%4==2) q1=min(q1,release[row+j*nf+g]);
        long used_other=0;
        for(long h=0;h<nf;h++) if(h!=f) {
          long processing=x[j*nf+h]+inc[row+j*nf+h]-(h==g?q1:0);
          used_other+=costs[(j*nf+h)*stride+processing];
        }
        long replicas=(ram[j]-used_other)/memory[f];
        long processing=x[j*nf+f]+inc[row+j*nf+f]-(f==g?q1:0);
        long q2=min(z[row+i*nf+f],max(0L,limits[(j*nf+f)*params[6]+replicas]-processing));
        edges[0]=other;delta[0]=-q1;edges[1]=target;delta[1]=q1;
        edges[2]=e;delta[2]=q2;
      }
    }
  } else {
    rng=(rng*16807)%2147483647L;
    long other=rng%ne;
    if(!((fun[other]==f && src[other]==i) || dst[other]==j)) {
      long target1=-1,target2=-1,origin=src[other],f2=fun[other];
      for(long a=starts[i*nf+f];a<starts[i*nf+f+1];a++)
        if(dst[a]==dst[other]) target1=a;
      for(long a=starts[origin*nf+f2];a<starts[origin*nf+f2+1];a++)
        if(dst[a]==j) target2=a;
      if(target1>=0 && target2>=0) {
        long q1=y[p*ne+e],q2=y[p*ne+other];
        if(c%2==0) {q1=min(q1,1L);q2=min(q2,1L);}
        edges[0]=e;delta[0]=-q1;edges[1]=target1;delta[1]=q1;
        edges[2]=other;delta[2]=-q2;edges[3]=target2;delta[3]=q2;
      }
    }
  }
}
for(long a=0;a<4;a++) {moves[idx*8+a*2]=edges[a];moves[idx*8+a*2+1]=delta[a];}
bool feasible=true;
for(long a=0;a<4 && feasible;a++) {
  long e=edges[a]; if(e<0) continue;
  long change=0;
  for(long b=0;b<4;b++) if(edges[b]==e) change+=delta[b];
  if(y[p*ne+e]+change<0) {feasible=false;break;}
  for(long endpoint=0;endpoint<2 && feasible;endpoint++) {
    long node=endpoint==0?src[e]:dst[e], used=0;
    for(long f=0;f<nf;f++) {
      long di=0,dout=0;
      for(long b=0;b<4;b++) {
        long other=edges[b];
        if(other>=0 && fun[other]==f) {
          if(src[other]==node) dout+=delta[b];
          if(dst[other]==node) di+=delta[b];
        }
      }
      long k=node*nf+f, received=inc[row+k]+di, rejected=z[row+k]-dout;
      long outgoing=load[k]-x[k]-rejected, processing=x[k]+received;
      if(received<0 || rejected<0 || outgoing<0 || (received>0 && outgoing>0)
          || processing<0 || processing>=stride) {feasible=false;break;}
      long cost=costs[k*stride+processing];
      if(cost>ram[node]-used) {feasible=false;break;}
      used+=cost;
    }
  }
}
valid[idx]=feasible;
