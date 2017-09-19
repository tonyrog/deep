%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    Random utilities
%%% @end
%%% Created : 15 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(deep_random).

-export([uniform/0, uniform/2]).
-export([shuffle/1]).

%% generate a double precision random number in [0-1)
uniform() ->
    <<_:4,X:52>> = crypto:strong_rand_bytes(7),
    <<F:64/float>> = <<16#3ff:12,X:52>>,
    F - 1.

uniform(Min, Max) ->
    Min + (uniform()*(Max - Min )).

shuffle(List) ->
    [Y || {_,Y} <- lists:sort([{uniform(),X} || X <- List])].
