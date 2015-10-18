%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2015, Tony Rogvall
%%% @doc
%%%    Multiply matrices using openCL
%%% @end
%%% Created : 27 Sep 2015 by Tony Rogvall <tony@rogvall.se>

-module(clmat).

-include("clmat.hrl").

-compile(export_all).

-define(MAX_REGS, 4).
-define(EPS, 0.01).

-record(mem,
	{
	  size :: unsigned(),
	  type :: atom(),          %% item type float,double,...
	  endian :: little | big,  %% item endian
	  object :: cl_mem()       %% cl object
	}).

-record(s,
	{
	  clu,         %% clu setup
	  queue,       %% queue
	  reg :: { #mem{} },
	  program,     %% mat.cl
	  mul_f,       %% reg[i] = reg[j]*reg[k]
	  add_f,       %% reg[i] = reg[j]+reg[k]
	  sub_f,       %% reg[i] = reg[j]-reg[k]
	  negate_f,    %% reg[i] = -reg[j]
	  sigmoid_f,    %% reg[i] = sigmoid(reg[j])
	  sigmoid_prime_f %% reg[i] = sigmoid(reg[j])*(1-sigmoid(reg[j]))
	}).

-spec multiply(A::#matrix{}, B::#matrix{}, Type::cl:cl_device_type()) ->
		      #matrix{}.

simple_multiply(Type) ->
    A = random(3, 3),
    B = random(3, 3),
    AL = matrix_to_lmat(A),
    BL = matrix_to_lmat(B),
    C = multiply(A, B, Type),
    compare_lmat(matrix_to_lmat(C), lmat:multiply(AL, BL)).

multiply(A, B, Type) when A#matrix.m =:= B#matrix.n ->
    N = max(max(A#matrix.n, A#matrix.m),max(B#matrix.n, B#matrix.m)),
    S = setup(N, Type),
    {ok,Ai,Es1} = write(A, 1, [],  S),
    {ok,Bi,Es2} = write(B, 2, Es1, S),
    Ci = #clmat { i=3,n=Ai#clmat.n,m=Bi#clmat.m,e=Ai#clmat.e},
    Es3 = mul_f(Ai, Bi, Ci, Es2, S),
    {ok,C} = read(Ci, Es3, S),
    C.

%% HasDouble = clu:devices_has_extension(Clu, "cl_khr_fp64"),

%% setup context/device/program/kernels/queue and buffers
setup(N,Type) ->
    Clu = clu:setup(Type),
    {ok,Q} = cl:create_queue(clu:context(Clu),clu:device(Clu),[]),
    Filename = filename:join(code:priv_dir(deep), "mat.cl"),
    Endian = little,
    ElemSize = 4,
    RegSz = N*N*ElemSize,  %% max matrix usage
    Reg = list_to_tuple(
	    [ begin
		  {ok,Buf} = cl:create_buffer(clu:context(Clu), 
					      [read_write], RegSz),
		  #mem { size = N*N, type = float,
			 endian = Endian, object = Buf }
	       end || _ <- lists:seq(1, ?MAX_REGS)]),
    {ok, Program} = clu:build_source_file(Clu, Filename, []),
    {ok, Mul_f} = cl:create_kernel(Program, "mul_f"),
    {ok, Add_f} = cl:create_kernel(Program, "add_f"),
    {ok, Sub_f} = cl:create_kernel(Program, "sub_f"),
    {ok, Negate_f} = cl:create_kernel(Program, "negate_f"),
    {ok, Sigmoid_f} = cl:create_kernel(Program, "sigmoid_f"),
    {ok, Sigmoid_prime_f} = cl:create_kernel(Program, "sigmoid_prime_f"),

    #s { clu = Clu, 
	 queue = Q,
	 reg = Reg,
	 program = Program,
	 mul_f = Mul_f, 
	 add_f = Add_f,
	 sub_f = Sub_f,
	 negate_f = Negate_f,
	 sigmoid_f = Sigmoid_f,
	 sigmoid_prime_f = Sigmoid_prime_f
       }.

-spec mul_f(A::#clmat{}, B::#clmat{}, C::#clmat{}, 
	    Es::[cl_event()], S::#s{}) -> [cl_event()].
mul_f(A, B, Dst, Es, S) ->
    binary_operation(#s.mul_f, A, B, Dst, Es, S).

-spec add_f(A::#clmat{}, B::#clmat{}, Dst::#clmat{},
	    Es::[cl_event()], S::#s{}) -> [cl_event()].
add_f(A, B, Dst, Es, S) ->
    binary_operation(#s.add_f, A, B, Dst, Es, S).

-spec sub_f(A::#clmat{}, B::#clmat{}, Dst::#clmat{},
	    Es::[cl_event()], S::#s{}) -> [cl_event()].
sub_f(A, B, Dst, Es, S) ->
    binary_operation(#s.sub_f, A, B, Dst, Es, S).

-spec sigmoid_f(A::#clmat{}, Dst::#clmat{},
		Es::[cl_event()], S::#s{}) -> [cl_event()].
sigmoid_f(A, Dst, Es, S) ->
    unary_operation(#s.sigmoid_f, A, Dst, Es, S).

sigmoid_prime_f(A, Dst, Es, S) ->
    unary_operation(#s.sigmoid_prime_f, A, Dst, Es, S).

unary_operation(Op,Src,Dst,Es,S) ->
    Rs = element(Src#clmat.i, S#s.reg),
    Rd = element(Dst#clmat.i, S#s.reg),
    Kernel = element(Op, S),
    clu:apply_kernel_args(Kernel,
			  [Src#clmat.m,
			   Rs#mem.object, Rd#mem.object]),
    Global = [Src#clmat.n,Src#clmat.m],
    Local = [],
    {ok,E} = cl:enqueue_nd_range_kernel(S#s.queue, Kernel,
					Global, Local, Es),
    [E|Es].

binary_operation(Op,Src1,Src2,Dst,Es,S) ->
    Rs1 = element(Src1#clmat.i, S#s.reg),
    Rs2 = element(Src2#clmat.i, S#s.reg),
    Rd = element(Dst#clmat.i, S#s.reg),
    Kernel = element(Op, S),
    clu:apply_kernel_args(Kernel,
			  [Src1#clmat.m,
			   Rs1#mem.object, Rs2#mem.object, Rd#mem.object]),
    Global = [Src1#clmat.n,Src1#clmat.m],
    Local = [],
    {ok,E} = cl:enqueue_nd_range_kernel(S#s.queue, Kernel,
					Global, Local, Es),
    [E|Es].
    

flush(Es, S) ->
    cl:flush(S#s.queue),
    cl:wait_for_events(Es).

%% write matrix A to register Ri
write(A, I, Es, S) ->
    Ri = element(I, S#s.reg),
    ElemSize = sizeof(A#matrix.type),
    Sz = A#matrix.n*A#matrix.m*ElemSize,
    {ok,E} = cl:enqueue_write_buffer(S#s.queue, Ri#mem.object, 0, Sz,
				     A#matrix.data, Es),
    {ok,#clmat{i=I,n=A#matrix.n,m=A#matrix.m,e=ElemSize},[E|Es]}.

%% read matrix A from register
read(A, Es, S) ->
    Ri = element(A#clmat.i, S#s.reg),
    Sz = A#clmat.n*A#clmat.m*A#clmat.e,
    {ok,E} = cl:enqueue_read_buffer(S#s.queue,Ri#mem.object,0,Sz,Es),
    {ok,Cdata} = cl:wait(E),
    cl:wait_for_events(Es),
    {ok,#matrix { n = A#clmat.n, m = A#clmat.m, type = float, data = Cdata }}.

sizeof(float) -> 4;
sizeof(double) -> 8.

matrix_to_lmat(#matrix { n=_N, m=M, type=double, data=Data }) ->
    R = M*8,
    [ [ E || <<E:64/little-float>> <= Row] || <<Row:R/binary>> <= Data ];
matrix_to_lmat(#matrix { n=_N, m=M, type=float, data=Data }) ->
    R = M*4,
    [ [ E || <<E:32/little-float>> <= Row] || <<Row:R/binary>> <= Data ].

lmat_to_matrix_32(A) ->
    N = length(A),
    M = length(hd(A)),
    R = M*4,
    Data = << << (<< <<E:32/float>> || E <- Row >>):R/binary>> || Row <- A >>,
    #matrix { n=N, m=M, type=float, data=Data }.

lmat_to_matrix_64(A) ->
    N = length(A),
    M = length(hd(A)),
    R = M*8,
    Data = << << (<< <<E:64/float>> || E <- Row >>):R/binary>> || Row <- A >>,
    #matrix { n=N, m=M, type=double, data=Data }.

compare(A, B) 
  when A#matrix.n =:= B#matrix.n,
       A#matrix.m =:= B#matrix.m ->
    LA = matrix_to_lmat(A),
    LB = matrix_to_lmat(B),
    compare_lmat(LA, LB).

compare_lmat([A|As], [B|Bs]) ->
    case lists:all(fun({Ai,Bi}) -> abs(Ai-Bi) < ?EPS end,
		   lists:zip(A,B)) of
	true -> compare_lmat(As, Bs);
	false -> false
    end;
compare_lmat([], []) ->
    true.


-spec random(N::unsigned(),M::unsigned()) -> #matrix{}.
random(N,M) ->
    #matrix { n=N, m=M, type=float, endian=little,
	      data= << << (frandom(32)):32/little-float >>|| _<-lists:seq(1,N*M) >> }.

%% generate a float precision random number in [0-1)
frandom(32) ->
    R = random(23),
    <<F:32/float>> = <<16#7F:9,R:23>>,
    F - 1;
frandom(64) ->
    R = random(52),
    <<F:64/float>> = <<16#3FF:12,R:52>>,
    F - 1.

%% generate N random bits
random(N) ->
    K = (N + 7) div 7,
    <<R:N,_/bitstring>> = crypto:rand_bytes(K),
    R.
