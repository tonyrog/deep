%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2015, Tony Rogvall
%%% @doc
%%%    Bench mark lmat
%%% @end
%%% Created : 22 Sep 2015 by Tony Rogvall <tony@rogvall.se>

-module(lmat_bench).

-compile(export_all).

test(N) -> test(N,1000).
test(N,L) ->
    ME = self(),
    Pid = 
	spawn(fun() ->
		      A = lmat:new(N, N),
		      B = lmat:new(N, N),
		      T0 = erlang:system_time(micro_seconds),
		      loop(L,A,B),
		      T1 = erlang:system_time(micro_seconds),
		      ME ! {self(),(L/(T1 - T0))*1000000}
	      end),
    receive
	{Pid,MultPerS} ->
	    MultPerS
    end.

test_p(N) -> test_p(N,1000).
test_p(N,L) ->
    ME = self(),
    Pid = 
	spawn(fun() ->
		      A = lmat:new(N, N),
		      B = lmat:new(N, N),
		      T0 = erlang:system_time(micro_seconds),
		      loop_p(L,A,B),
		      T1 = erlang:system_time(micro_seconds),
		      ME ! {self(),(L/(T1 - T0))*1000000}
	      end),
    receive
	{Pid,MultPerS} ->
	    MultPerS
    end.

	
loop(0, _A, _B) ->
    ok;
loop(I, A, B) ->
    lmat:multiply(A, B),
    loop(I-1, A, B).

loop_p(0, _A, _B) ->
    ok;
loop_p(I, A, B) ->
    lmat:multiply_p(A, B),
    loop_p(I-1, A, B).
