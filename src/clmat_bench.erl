%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2015, Tony Rogvall
%%% @doc
%%%    Benchmark cl matrix 
%%% @end
%%% Created :  1 Oct 2015 by Tony Rogvall <tony@rogvall.se>

-module(clmat_bench).

-compile(export_all).

-include("clmat.hrl").


test(N) -> test(N, 1000, gpu).
test(N,Type) -> test(N,1000,Type).

test(N,L,Type) ->
    ME = self(),
    Pid =
	spawn(fun() ->
		      A = clmat:random(N, N),
		      B = clmat:random(N, N),
		      S = clmat:setup(N, Type),
		      {ok,Ai,Es1} = clmat:write(A, 1, [],  S),
		      {ok,Bi,Es2} = clmat:write(B, 2, Es1, S),
		      clmat:flush(Es2, S),
		      T0 = erlang:system_time(micro_seconds),
		      loop(L,Ai,Bi,S),
		      T1 = erlang:system_time(micro_seconds),
		      ME ! {self(),(L/(T1 - T0))*1000000}
	      end),
    receive
	{Pid,MultPerS} ->
	    N = N, M = N, P = N,
	    {MultPerS, trunc((MultPerS*(N*M*P))/1000000)}
    end.

loop(0, _A, _B,_S) ->
    ok;
loop(I,A,B,S) ->
    C = #clmat { i=3,n=A#clmat.n,m=B#clmat.m,e=A#clmat.e},
    Es = clmat:mul_f(A, B, C, [], S),
    clmat:flush(Es, S),
    loop(I-1, A, B, S).
