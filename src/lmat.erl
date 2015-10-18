%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2015, Tony Rogvall
%%% @doc
%%%    Matrix as list of lists
%%% @end
%%% Created : 20 Sep 2015 by Tony Rogvall <tony@rogvall.se>

-module(lmat).

-export([multiply/2]).
-export([transpose/1]).

-compile(export_all).
%% -compile(native).

-type vector() :: [number()].
-type matrix() :: [vector()].
-type unsigned() :: non_neg_integer().

-spec new(N::integer(), M::integer()) -> matrix().

new(N,M) ->
    [ randomv(M) || _ <- lists:seq(1, N)].
%%
%%  [[1 2],   [[5,6],     [[1*5+2*7,1*6+2*8],   [[19,22],
%%          *      =                          =   
%%   [3,4]]    [7,8]]      [3*5+4*7,3*6+4*8]]    [43,50]]
%%

-spec multiply(A::matrix(), B::matrix()) -> matrix().
multiply(A, B) ->
    multiply_t(A, transpose(B)).

%% multipy by transpose B matrix
multiply_t(A, Bt) ->
    [ [ dotv(Bi,Ai) || Bi <- Bt] || Ai <- A ].


-spec multiply_p(A::matrix(), B::matrix()) -> matrix().
multiply_p(A, B) ->
    multiply_tp(A, transpose(B)).

%% multipy by transpose B matrix
multiply_tp(A, Bt) ->
    parlists:map(fun(Ai) -> 
			 lists:map(fun(Bi) -> dotv(Bi,Ai) end, Bt)
		 end, A).

-spec add(A::matrix(), B::matrix()) -> matrix().
add([Ai|As], [Bi|Bs]) -> [addv(Ai,Bi) | add(As,Bs)];
add([], []) -> [].

addv([A|As],[B|Bs]) -> [A+B | addv(As,Bs)];
addv([],[]) -> [].

-spec subtract(A::matrix(), B::matrix()) -> matrix().
subtract([Ai|As], [Bi|Bs]) -> [subv(Ai,Bi) | subtract(As,Bs)];
subtract([], []) -> [].

subv([A|As],[B|Bs]) -> [A-B | subv(As,Bs)];
subv([],[]) -> [].

%% Vector dot product
dotv(Xs, Ys) ->
    dotv_(Xs, Ys, 0).

dotv_([Xi|Xs], [Yi|Ys], D) ->
    dotv_(Xs, Ys, D+Xi*Yi);
dotv_([], [], D) -> D.

%% Transpose a matrix
%%    [[5,6]      [[5,7],
%%             = 
%%     [7,8]]      [6,8]]
transpose([Ri]) ->
    [ [Xj] || Xj <- Ri];
transpose([Ri|Rs]) ->
    cons_zip(Ri, transpose(Rs)).

cons_zip([Xj|Xjs], [Ri|Rs]) ->
    [[Xj|Ri] | cons_zip(Xjs,Rs)];
cons_zip([], []) ->
    [].



-spec randomv(N::unsigned()) -> vector().
randomv(N) ->
    [ frandom(64) || _ <- lists:seq(1,N)].

-spec randomv(N::unsigned(), Min::number(), Max::number()) -> vector().
randomv(N, Min, Max) ->
    [ Min + frandom(64)*(Max - Min) || _ <- lists:seq(1,N)].
    
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
