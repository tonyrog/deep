%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    deep_net
%%% @end
%%% Created : 14 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(deep_net).

-export([new/3]).
-export([feed/2]).
-export([sgd/4]).

-compile(export_all).

-define(DEFAULT_ACTIVATION, sigmoid).

-record(layer,
	{
	  inputs  :: integer(),         %% number of inputs
	  outputs  :: integer(),        %% number of outputs
	  weights :: matrix:matrix(),   %% weight matrix
	  bias    :: matrix:matrix(),   %% bias vector
	  locked = false :: boolean(),  %% update weights?
	  activation :: atom(),         %% name of activation function
	  activation_fn :: fun((I::matrix:matrix()) -> 
				      matrix:matrix()),
	  gradient_fn :: fun((I::matrix:matrix(),O::matrix:matrix()) -> 
				    matrix:matrix())
	}).

%% create a network with an input vector of size Ni 
%% then a list of sizes of the hidden layers, lastly the
%% output vector.
%%
new(IN, Hs, OUT) when is_integer(IN), IN > 0,
		      is_list(Hs) ->
    create__(IN, Hs++[OUT]).

create__(IN,[OUTL|Hs]) ->
    {Activation,OUT} =
	case OUTL of
	    {Act,Out} when is_atom(Act), is_integer(Out), Out > 0 ->
		{Act,Out};
	    Out when is_integer(Out), Out > 0 ->
		{?DEFAULT_ACTIVATION, Out}
	end,
    %% W = matrix:normal(OUT,IN,float32),  %% weights hidden layers
    Ws = [ matrix:to_list(matrix:scale(1/math:sqrt(IN),matrix:normal(1,IN,float32))) || _ <- lists:seq(1,OUT)],
    W  = matrix:from_list(lists:append(Ws),float32),
    B = matrix:normal(OUT,1,float32),   %% bias vector
    F = activation_function(Activation),
    G = gradient_function(Activation),
    L = #layer { inputs = IN,
		 outputs = OUT,
		 weights = W,
		 bias = B,
		 activation = Activation,
		 activation_fn = F,
		 gradient_fn = G
	       },
    [L | create__(OUT,Hs)];
create__(_IN,[]) ->
    [].

%%
%% Feed forward:
%%
feed(Net,IN) ->
    feed_(Net,IN).

feed_([#layer{weights=W,bias=B,activation_fn=F}|Net],A) ->
    Z = matrix:add(matrix:multiply(W,A), B),
    feed_(Net, F(Z));
feed_([], A) ->
    A.

sgd(Net,TrainingSet,ValidationSet,InputOptions)
      when is_list(TrainingSet),
	   is_list(ValidationSet),
	   is_map(InputOptions) ->
    Default = #{ eta => 3.0, 
		 lmbda => 0.0,
		 batch_size => 10,
		 epochs => 20,
		 k => 0,
		 learn => learn  %% with batch
	       },
    Options = maps:merge(Default, InputOptions),
    Eta   = option(learning_rate, Options),
    Lmbda = option(regularization, Options),
    BatchSize = option(batch_size, Options),
    Epochs = option(epochs, Options),
    K = option(k, Options),
    L = option(learn, Options),
    Tn = length(TrainingSet),
    sgd_(Net,L,Tn,TrainingSet,ValidationSet,1,Epochs,BatchSize,Eta,Lmbda,K).

sgd_(Net,_L,_Tn,_TrainingSet,_ValidationSet,E,Epochs,_BatchSize,_Eta,_Lmbda,_K)
  when E > Epochs -> Net;
sgd_(Net,L,Tn,TrainingSet,ValidationSet,E,Epochs,BatchSize,Eta,Lmbda,K) ->
    TrainingSet1 = deep_random:shuffle(TrainingSet),
    T0 = erlang:monotonic_time(),
    Net1 = sgd__(Net,L,Tn,TrainingSet1,BatchSize,Eta,Lmbda,K),
    T1 = erlang:monotonic_time(),
    R = evaluate(Net1,ValidationSet),
    T2 = erlang:monotonic_time(),
    M = length(ValidationSet),
    Time1 = erlang:convert_time_unit(T1 - T0, native, microsecond),
    Time2 = erlang:convert_time_unit(T2 - T1, native, microsecond),
    io:format("epoch ~w/~w : learn=~.2f%, train=~.2fs, eval=~.2fs\n", 
	      [E,Epochs,(R/M)*100,Time1/1000000,Time2/1000000]),
    sgd_(Net1,L,Tn,TrainingSet,ValidationSet,E+1,Epochs,BatchSize,Eta,Lmbda,K).

%% loop over batches and train the net
sgd__(Net,_L, _Tn, [], _BatchSize, _Eta, _Lmbda, _K) ->
    Net;
sgd__(Net, L, Tn,TrainingSet, BatchSize, Eta, Lmbda, K) ->
    Batch = lists:sublist(TrainingSet, BatchSize),
    Bn = length(Batch),
    {Ws,Bs} = case L of 
		  learn -> learn(Net, Batch, K);
		  learn1 -> learn1(Net, Batch, K);
		  plearn -> plearn(Net, Batch, K)
	      end,
    Net1 = update_net(Net, Ws, Bs, Eta, Bn, Lmbda, Tn),
    TrainingSet1 = lists:nthtail(Bn, TrainingSet),
    sgd__(Net1, L, Tn, TrainingSet1, BatchSize, Eta, Lmbda, K).


%%
%% parallel learn
%%
plearn(Net, Batch, K) ->
    WsBs = parlists:map(
	     fun({X,Y}) ->
		     backprop(Net, X, Y, K)
	     end, Batch),
    sum_delta(WsBs).

sum_delta([{Ws,Bs}|WsBs]) ->
    sum_delta(WsBs, Ws, Bs).

sum_delta([{Ws,Bs}|WsBs], Ws0, Bs0) ->
    Ws1 = lists:zipwith(fun matrix:add/2, Ws, Ws0),
    Bs1 = lists:zipwith(fun matrix:add/2, Bs, Bs0),
    sum_delta(WsBs, Ws1, Bs1);
sum_delta([], Ws, Bs) ->
    {Ws,Bs}.

%%
%% Run learning over a batch of {X,Y} pairs
%% This mean update the weights after each batch
%%
learn(Net, [{X,Y}|Batch], K) ->
    {Ws,Bs} = backprop(Net, X, Y, K),
    learn(Net, Batch, Ws, Bs, K).

learn(Net, [{X,Y}|Batch], Ws, Bs, K) ->
    {WsB,BsB} = backprop(Net, X, Y, K),
    Ws1 = lists:zipwith(fun matrix:add/2, Ws, WsB),
    Bs1 = lists:zipwith(fun matrix:add/2, Bs, BsB),
    learn(Net, Batch, Ws1, Bs1, K);
learn(_Net, [], Ws, Bs, _K) ->
    {Ws, Bs}.

%% Bn = mini batch size
%% Tn = training set size
update_net([L=#layer{locked=true}|Net],[_NW|Ws],[_NB|Bs],Eta,Bn,Lmbda,Tn) ->
    [L | update_net(Net, Ws, Bs, Eta, Bn, Lmbda, Tn)];
update_net([L=#layer{weights=W,bias=B}|Net],[NW|Ws],[NB|Bs],Eta,Bn,Lmbda,Tn) ->
    W1 = matrix:subtract(matrix:scale((1-Eta*Lmbda/Tn),W),
			 matrix:scale(Eta/Bn,NW)),
    B1 = matrix:subtract(B, matrix:scale(Eta/Bn, NB)),
    [L#layer{weights=W1,bias=B1} | update_net(Net,Ws,Bs,Eta,Bn,Lmbda,Tn)];
update_net([], [], [], _Eta, _Bn, _Lmbda, _Tn) ->
    [].

%%
%% back propagation algorithm,
%% adjust weights and biases to match output Y on input X
%%
backprop(Net=[#layer{gradient_fn=G}|Net1], X, Y, K) ->
    {[Out|As=[A0|_]],[Z|Zs]} = feed_forward_(Net, X, [X], []),
    Grad = G(Z,Out),
    Ki = matrix:topk(Grad,K),
    Delta = matrix:ktimes(cost_derivative(Out, Y), Grad, Ki),
    W1 = matrix:kmultiply(Delta,matrix:transpose(A0),Ki),
    NetR = lists:reverse(Net1),
    backward_pass_(NetR,Delta,As,Zs,[W1],[Delta],K).

backward_pass_([#layer{weights=W,bias=_B,gradient_fn=G}|Net],
	       Delta0,[Out|As=[A0|_]],[Z|Zs],Nw,Nb,K) ->
    Grad = G(Z,Out),
    Ki     = matrix:topk(Grad,K),
    DeltaC = matrix:transpose(matrix:transpose_data(Delta0)), %% column order
    Wt     = matrix:transpose(W),
    WtD    = matrix:kmultiply(Wt, DeltaC, Ki),
    Delta  = matrix:ktimes(WtD, Grad, Ki),
    W1     = matrix:kmultiply(Delta,matrix:transpose(A0),Ki),
    backward_pass_(Net,Delta,As,Zs,[W1|Nw],[Delta|Nb],K);
backward_pass_([],_,_,_,Nw,Nb,_) ->
    {Nw,Nb}.

%%
%% variant of learn
%%
learn1(Net, [{X,Y}|Batch], K) ->
    {Ws, Bs} = learn1_(Net, X, Y, K),
    learn1(Net, Batch, Ws, Bs, K).

learn1(Net, [{X,Y}|Batch], Ws0, Bs0, K) ->
    {Ws, Bs} = learn1_(Net, X, Y, K),
    Ws1 = lists:zipwith(fun matrix:add/2, Ws, Ws0),
    Bs1 = lists:zipwith(fun matrix:add/2, Bs, Bs0),
    learn1(Net, Batch, Ws1, Bs1, K);
learn1(_Net, [], Ws, Bs, _K) ->
    {Ws,Bs}.
    
learn1_([#layer{weights=W,bias=B,activation_fn=F,gradient_fn=G}], A, Y, K) ->
    %% output layer
    Z = matrix:add(matrix:multiply(W, A), B),
    Out = F(Z),
    Grad = G(Z,Out),
    Ki = matrix:topk(Grad,K),
    Delta = matrix:ktimes(cost_derivative(Out, Y), Grad, Ki),
    W1 = matrix:kmultiply(Delta,matrix:transpose(A),Ki),
    {[W1],[Delta]};
learn1_([#layer{weights=W,bias=B,activation_fn=F,gradient_fn=G}|
	 Net=[#layer{weights=Wn}|_]], A, Y, K) ->
    %% hidden layer
    Z = matrix:add(matrix:multiply(W,A),B),
    Out = F(Z),
    {Ws,Bs=[Delta0|_]} = learn1_(Net,Out,Y,K),
    Grad = G(Z,Out),
    Ki = matrix:topk(Grad,K),
    DeltaC = matrix:transpose(matrix:transpose_data(Delta0)),
    Wt = matrix:transpose(Wn), %% Note Wn!!!
    WtD = matrix:kmultiply(Wt, DeltaC, Ki),
    Delta = matrix:ktimes(WtD, Grad, Ki),
    W1 = matrix:kmultiply(Delta,matrix:transpose(A),Ki),
    {[W1|Ws],[Delta|Bs]}.

%% more cost functions
cost_derivative(A, Y) ->
    matrix:subtract(A, Y).

%% Forward and store Z in a list
feed_forward_([#layer{weights=W,bias=B,activation_fn=F}|Net], A, As, Zs) ->
    Z = matrix:add(matrix:multiply(W, A), B),
    A1 = F(Z),
    feed_forward_(Net, A1, [A1|As], [Z|Zs]);
feed_forward_([], _A, As, Zs) ->
    {As, Zs}.

evaluate(Net, TestData) ->
    evaluate_(Net, TestData, 0).

evaluate_(Net, [{X,Y}|TestData], Sum) ->
    Y1 = feed(Net, X),
    Yi = matrix:element(1,1,matrix:argmax(Y1,0))-1,
    %% io:format("Y = ~w, Yi = ~w\n", [Y, Yi]),
    if Y =:= Yi -> evaluate_(Net, TestData, Sum+1);
       true -> evaluate_(Net, TestData, Sum)
    end;
evaluate_(_Net, [], Sum) ->
    Sum.

%% options
option(Key, Options) ->
    option(Key, Options, undefined).

option(Key, Options, Default) ->
    case maps:find(Key, Options) of
	{ok,Value} -> validate(Key, Value);
	%% check aliases
	error when Key =:= learning_rate -> option_(eta, Options, Default);
	error when Key =:= eta -> option_(learning_rate, Options, Default);
	error when Key =:= regularization ->  option_(lmbda, Options, Default);
	error when Key =:= lmbda -> option_(regularization, Options, Default);
	error -> validate(Key, Default)
    end.

option_(Key, Options, Default) ->
    validate(Key, maps:get(Key, Options, Default)).

validate(k, Value) when is_integer(Value), Value >= 0 -> Value;
validate(eta, Value) when is_number(Value), Value >= 0 -> Value;
validate(learning_rate, Value) when is_number(Value), Value >= 0 -> Value;
validate(lmbda, Value) when is_number(Value), Value >= 0 -> Value;
validate(regularization, Value) when is_number(Value), Value >= 0 -> Value;
validate(epochs, Value) when is_integer(Value), Value >= 0 -> Value;
validate(batch_size, Value) when is_integer(Value), Value >= 0 -> Value;
validate(learn, learn) -> learn;
validate(learn, learn1) -> learn1;
validate(learn, plearn) -> plearn.

activation_function(sigmoid) -> fun matrix:sigmoid/1;
activation_function(tanh) ->  fun matrix:tanh/1;
activation_function(relu) -> fun matrix:relu/1;
activation_function(leaky_relu) -> fun matrix:leaky_relu/1;
activation_function(linear) -> fun matrix:linear/1;
activation_function(softplus) -> fun matrix:softplus/1.

gradient_function(sigmoid) -> fun matrix:sigmoid_prime/2;
gradient_function(tanh) ->  fun matrix:tanh_prime/2;
gradient_function(relu) -> fun matrix:relu_prime/2;
gradient_function(leaky_relu) -> fun matrix:leaky_relu_prime/2;
gradient_function(linear) -> fun matrix:linear_prime/2;
gradient_function(softplus) -> fun matrix:softplus_prime/2.
    
%% dump network to terminal for inspection
dump(Net) ->
    dump(1, Net).
    
dump(I,[L=#layer { inputs=N, outputs=M, activation=A } | Ls]) ->
    if I =:= 1 ->
	    io:format("input ~s layer ~wx~w\n", [A, N, M]);
       Ls =:= [] ->
	    io:format("output ~s layer ~wx~w\n", [A, N, M]);
       true ->
	    io:format("hidden ~s layer ~wx~w\n", [A, N, M])
    end,
    io:format(" WEIGHTS\n"),
    matrix:print(L#layer.weights),
    io:format(" BIAS\n"),
    matrix:print(L#layer.bias),
    dump(I+1,Ls);
dump(_, []) ->
    ok.



    
    
    

