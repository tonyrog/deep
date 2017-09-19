%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    deep_net
%%% @end
%%% Created : 14 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(deep_net).

-export([new/3]).
-export([feed/2]).
-export([backprop/3]).
-export([learn/3]).
-export([sgd/5, sgd/6]).

-compile(export_all).

%% create a network with an input vector of size Ni 
%% then a list of sizes of the hidden layers, lastly the
%% output vector.
%%
new(Ni, Hs, No) when is_integer(Ni), Ni > 0,
		     is_list(Hs),
		     is_integer(No), No > 0 ->
    create__(Ni, Hs++[No]).

create__(Nh,[Nhh|Hs]) ->
    W = matrix:normal(Nhh,Nh,float32),  %% weights hidden layers
    B = matrix:normal(Nhh,1,float32),   %% bias vector
    [{W,B} | create__(Nhh,Hs)];
create__(_Nh,[]) ->
    [].

%%
%% Feed forward 
%%
feed(Net,IN) ->
    feed_(Net,IN).

feed_([{W,B}|Net],A) ->
    Z = matrix:add(matrix:multiply(W,A), B),
    feed_(Net, matrix:sigmoid(Z));
feed_([], A) ->
    A.

sgd(Net, TrainingSet, ValidationSet, Epocs, BatchSize, Rate) ->
    sgd_(Net, TrainingSet, ValidationSet, Epocs, BatchSize, Rate).

sgd(Net, TrainingSet, Epocs, BatchSize, Rate) 
  when is_list(TrainingSet),
       is_integer(Epocs), Epocs > 0,
       is_integer(BatchSize),
       is_number(Rate), Rate < 1 ->
    sgd_(Net, TrainingSet, [], Epocs, BatchSize, Rate).

sgd_(Net, _TrainingSet, _ValidationSet, 0, _BatchSize, _Rate) ->
    Net;
sgd_(Net, TrainingSet, ValidationSet, Epocs, BatchSize, Rate) ->
    TrainingSet1 = deep_random:shuffle(TrainingSet),
    N = length(TrainingSet1),
    Net1 = sgd__(Net, TrainingSet1, N, BatchSize, Rate),
    R = evaluate(Net1, ValidationSet),
    M = length(ValidationSet),
    io:format("epoch remain ~w : ~w / ~w\n", [Epocs,R,M]),
    sgd_(Net1, TrainingSet, ValidationSet, Epocs-1, BatchSize, Rate).

%% loop over batches and train the net
sgd__(Net, TrainingSet, N, BatchSize, Rate) when BatchSize =< N ->
    {Batch,TrainingSet1} = lists:split(BatchSize, TrainingSet),
    %% io:format("learn\n"),
    Net1 = learn(Net, Batch, Rate),
    sgd__(Net1, TrainingSet1, N-BatchSize, BatchSize, Rate);
sgd__(Net, _TrainingSet, _N, _BatchSize, _Rate) ->
    Net.

%%
%% Run learning over a batch of {X,Y} pairs
%%
learn(Net, [{X,Y}|Batch], Eta) ->
    {DNw,DNb} = backprop(Net, X, Y),
    learn(Net, Batch, Eta/length(Batch), DNw, DNb).

learn(Net, [{X,Y}|Batch], Eta, Nw, Nb) ->
    {DNw,DNb} = backprop(Net, X, Y),
    Nw2 = add_lists(Nw, DNw),
    Nb2 = add_lists(Nb, DNb),
    learn(Net, Batch, Eta, Nw2, Nb2);
learn(Net, [], Eta, Nw, Nb) ->
    learn_net(Net, Nw, Nb, Eta).

learn_net([{W,B}|Net], [NW|Ws], [NB|Bs], Eta) ->
    W1 = matrix:subtract(W, matrix:scale(Eta, NW)),
    B1 = matrix:subtract(B, matrix:scale(Eta, NB)),
    [{W1,B1} | learn_net(Net, Ws, Bs, Eta)];
learn_net([], [], [], _Eta) ->
    [].
    
add_lists([A|As],[B|Bs]) ->
    add_lists_(As, Bs, [matrix:add(A,B)]).

add_lists_([A|As],[B|Bs],Cs) ->
    add_lists_(As,Bs,[matrix:add(A,B)|Cs]);
add_lists_([],[],Cs) ->
    lists:reverse(Cs).

%%
%% back propagation algorithm,
%% adjust weights and biases to match output Y on input X
%%
backprop(Net, X, Y) ->
    {[A1,A2|As],[Z|Zs]} = feed_forward_(Net, X, [X], []),
    SP =  matrix:sigmoid_prime(Z),
    D0 = matrix:times(cost_derivative(A1, Y), SP),
    NetR = lists:reverse(tl(Net)),
    backward_pass_(NetR,D0,As,Zs,[matrix:multiply(D0,matrix:transpose(A2))],[D0]).

backward_pass_([{W,_B}|Net],D0,[A|As],[Z|Zs],Nw,Nb) ->
    SP = matrix:sigmoid_prime(Z),
    Wt = matrix:transpose(W),
    WtD0 = matrix:multiply(Wt, D0),
    D1 = matrix:times(WtD0, SP),
    backward_pass_(Net,D1,As,Zs,
		   [matrix:multiply(D1,matrix:transpose(A))|Nw],[D1|Nb]);
backward_pass_([],_,_,_,Nw,Nb) ->
    {Nw,Nb}.

cost_derivative(A, Y) ->
    matrix:subtract(A, Y).

%% Forward and store Z in a list
feed_forward_([{W,B}|Net], A, As, Zs) ->
    Z = matrix:add(matrix:multiply(W, A), B),
    A1 = matrix:sigmoid(Z),
    feed_forward_(Net, A1, [A1|As], [Z|Zs]);
feed_forward_([], _A, As, Zs) ->
    {As, Zs}.

evaluate(Net, TestData) ->
    evaluate_(Net, TestData, 0).

evaluate_(Net, [{X,Y}|TestData], Sum) ->
    Y1 = feed(Net, X),
    Yi = hd(matrix:argmax(matrix:transpose(Y1))),
    %% io:format("Y = ~w, Yi = ~w\n", [Y, Yi]),
    if Y =:= Yi -> evaluate_(Net, TestData, Sum+1);
       true -> evaluate_(Net, TestData, Sum)
    end;
evaluate_(_Net, [], Sum) ->
    Sum.
