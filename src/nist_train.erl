%%% @author Tony Rogvall <tony@rogvall.se>
%%% @copyright (C) 2017, Tony Rogvall
%%% @doc
%%%    Train a neural network with the NIST data
%%% @end
%%% Created : 19 Aug 2017 by Tony Rogvall <tony@rogvall.se>

-module(nist_train).

-compile(export_all).

%% read the NIST data in two parts
%% 50000 train data and 10000 validation data
%%
main() ->
    main( #{ epochs => 20,
	     batch_size => 10,
	     learning_rate => 3.0 }).

main(Options) ->
    main([{sigmoid,30}], sigmoid, Options, 10000, 1000).

main_relu() ->
    main_relu( #{ epochs => 20,
		  batch_size => 10,
		  learning_rate => 3.0 }).

main_relu(Options) ->
    main([{relu,30}], sigmoid, Options, 10000, 1000).


main_large() ->
    main_large(#{epochs => 30,
		 batch_size=>10,
		 learning_rate=>3.0}).

main_large(Options) ->
    main([{sigmoid,30}], sigmoid, Options, 50000, 10000).

%% try divide and conquer
main_fed() ->
    main_fed(20).
main_fed(N) ->
    main_fed(N, #{epochs => 10,
		  batch_size=>10,
		  max_learn_factor => 0.96,
		  learning_rate=>3.0
		 }).
    
main_fed(N, Options) ->
    TTrain = 10000,
    TValid = 1000,
    NTrain = TTrain div N,
    NValid = TValid div N,
    Vt = 0,
    Vi = TTrain,
    Hidden =  [{sigmoid,30}],
    Otype  = sigmoid,
    NetList = [deep_net:new(784,Hidden,{Otype,10}) || _ <- lists:seq(1, N)],
    FTrain = fun(_Data) -> true end,
    FValid = fun(_Data) -> true end,
    Training = load_fed_1valid(N, Vt, NTrain, FTrain, Vi, NValid, FValid),
    %% Training = load_fed(N, Vt, NTrain, FTrain, Vi, NValid, FValid),
    fed_loop(NetList,1,Options,Training,Training,1.0,[]).

%%
%% load N training sets and N validation sets
%%
load_fed(N, Vt, NTrain, FTrain, Vi, NValid, FValid) ->
    load_fed(N, Vt, NTrain, FTrain, Vi, NValid, FValid, []).

load_fed(0, _Vt, _NTrain, _FTrain, _Vi, _NValid, _FValid, Sets) ->
    Sets;
load_fed(I, Vt, NTrain, FTrain, Vi, NValid, FValid, Sets) ->
    Train0 = load(Vt, NTrain, FTrain, fun nist:image_to_vector/1),
    Valid0 = load(Vi, NValid, FValid, fun nist:image_to_vector/1),
    Train = [ {X,nist:label_to_matrix(Y)} || {X,Y} <- Train0],
    Valid = [ {X,Y+1} || {X,Y} <- Valid0],
    load_fed(I-1, Vt+NTrain, NTrain, FTrain, 
	     Vi+NValid, NValid, FValid, [{Train,Valid}|Sets]).
%%
%% load N training sets but only one validation set
%%
load_fed_1valid(N, Vt, NTrain, FTrain, Vi, NValid, FValid) ->
    Valid = load(Vi, NValid, FValid, fun nist:image_to_vector/1),
    load_fed_1valid(N, Vt, NTrain, FTrain, Valid, []).

load_fed_1valid(0, _Vt, _NTrain, _FTrain, _Valid, Sets) ->
    Sets;
load_fed_1valid(I, Vt, NTrain, FTrain, Valid, Sets) ->
    Train0 = load(Vt, NTrain, FTrain, fun nist:image_to_vector/1),
    Train = [ {X,nist:label_to_matrix(Y)} || {X,Y} <- Train0],
    load_fed_1valid(I-1,Vt+NTrain,NTrain,FTrain,Valid,[{Train,Valid}|Sets]).

%% try divide and conquer digit by digit

main_digit_fed() ->
    main_digit_fed(#{epochs => 1,
		     batch_size => 10,
		     max_learn_factor => 0.60,
		     learning_rate=>4.0}).
    
main_digit_fed(Options) ->
    N = 5,
    TTrain = 10000,
    TValid = 1000,
    NTrain = TTrain div N,
    NValid = TValid div N,
    Vt = 0,
    Vi = TTrain,
    Hidden =  [{sigmoid,30}],
    Otype  = sigmoid,
    NetList = [deep_net:new(784,Hidden,{Otype,10}) || _ <- lists:seq(1, N)],
    Training = load_digit_fed(2*N, Vt, NTrain, Vi, NValid),
    fed_loop(NetList, 1, Options, Training,Training,1.0,[]).

%% load set training each digit but validate for anything
%% not 2 digits at a time
load_digit_fed(N, Vt, NTrain, Vi, NValid) ->
    FValid = fun(_Data) -> true end,
    Valid0 = load(Vi, NValid, FValid, fun nist:image_to_vector/1),
    Valid = [ {X,Y+1} || {X,Y} <- Valid0],
    load_digit_fed_(0, N, Vt, NTrain, Valid, []).

load_digit_fed_(I, N, _Vt, _NTrain, _Valid, Sets) when I >= N ->
    Sets;
load_digit_fed_(I, N, Vt, NTrain, Valid, Sets) ->
    FTrain = fun({_Img,L}) -> (L =:= I) orelse (L =:= I+1) end,
    Train0 = load(Vt, NTrain, FTrain, fun nist:image_to_vector/1),
    Train = [ {X,nist:label_to_matrix(Y)} || {X,Y} <- Train0],
    load_digit_fed_(I+2,N, Vt+NTrain, NTrain, 
		    Valid, [{Train,Valid}|Sets]).

fed_loop([Net|Ns], Round, Options, [{Train,Valid}|Sets], Sets0, MinLf, Acc) ->
    {Lf,Net1} = deep_net:sgdf(Net, Train, Valid, Options),
    fed_loop(Ns, Round, Options, Sets, Sets0, min(Lf,MinLf), [Net1|Acc]);
fed_loop([], Round, Options, _Sets, Sets0, MinLf, Acc) ->
    MaxLearnFactor = maps:get(max_learn_factor, Options),
    if MinLf > MaxLearnFactor ->
	    deep_net:avg_netlist(Acc);
       true ->
	    NetList = combine_networks(Acc),
	    io:format("round: ~w\n", [Round]),
	    fed_loop(NetList, Round+1, Options, Sets0, Sets0, 1.0, [])
    end.

combine_networks(NetList) ->
    Avg = deep_net:avg_netlist(NetList),
    N = length(NetList),
    [Avg | [deep_net:copy_netlist(Avg) || _ <- lists:seq(1,N-1)]].

main(Hidden, Otype, Options, NTrain, NValid) ->
    FTrain = fun(_Data) -> true end,
    FValid = fun(_Data) -> true end,
    Train0 = load(0, NTrain, FTrain, fun nist:image_to_vector/1),
    Valid0 = load(NTrain, NValid, FValid, fun nist:image_to_vector/1),
    Train = [ {X,nist:label_to_matrix(Y)} || {X,Y} <- Train0],
    Valid = [ {X,Y+1} || {X,Y} <- Valid0],
    io:format("loaded ~w+~w\n", [NTrain, NValid]),
    Net = deep_net:new(784,Hidden,{Otype,10}),
    deep_net:sgd(Net, Train, Valid, Options).

load2d(N) ->
    load(0,N,fun (_Data) -> true end,fun nist:image_to_matrix/1).

loadv(N) ->
    load(0,N,fun (_Data) -> true end,fun nist:image_to_vector/1).

%% load one set of test data
load(I,N,Filter,Transform) ->
    {ok,Fd1} = nist:open_image_file(),
    {ok,Fd2} = nist:open_label_file(),
    R = load_(Fd1,Fd2,I,N,Filter,Transform),
    nist:close_image_file(Fd1),
    nist:close_label_file(Fd2),
    R.

%% Traansform (Binary) -> matrix|vector
%% Filter ({TransformedImage, Label}) -> boolean()

load_(Fd1, Fd2, I, N, Filter, Transform) ->
    {ok,Img} = nist:read_image(Fd1, I),
    {ok,<<L>>} = nist:read_label(Fd2, I),
    First = { Transform(Img), L },
    case Filter(First) of
	false ->
	    load__(Fd1,Fd2, N, Filter, Transform, []);
	true ->
	    load__(Fd1,Fd2, N-1, Filter,  Transform, [First])
    end.

load__(Fd1,Fd2,N,Filter,Transform,Acc) when N > 0->
    {ok,Img} = nist:read_next_image(Fd1),
    {ok,<<L>>} = nist:read_next_label(Fd2),
    Next = { Transform(Img), L },
    case Filter(Next) of
	false ->
	    load__(Fd1,Fd2,N,Filter,Transform,Acc);
	true ->
	    load__(Fd1,Fd2,N-1,Filter,Transform,[Next|Acc])
    end;
load__(_Fd1,_Fd2,0,_Filter,_Transform, Acc) ->
    lists:reverse(Acc).

%% Test Net for mixed images !!!
%% return hit fraction

test_mix(Net) ->
    {ok,Fd1} = nist:open_image_file(),    
    {ok,Fd2} = nist:open_label_file(),
    K = test_mix_loop(Net, Fd1, Fd2, 1000, 0),
    (K/1000).

test_mix_loop(_Net, _Fd1, _Fd2, 0, N) ->
    N;
test_mix_loop(Net, Fd1, Fd2, L, N) ->
    I = rand:uniform(10000)-1,
    J = rand:uniform(10000)-1,
    if I =:= J ->
	    test_mix_loop(Net,Fd1,Fd2,L,N);
       true ->
	    {ok,Mi} = nist:read_image(Fd1, I),
	    {ok,<<Li>>} = nist:read_label(Fd2, I),
	    {ok,Mj} = nist:read_image(Fd1, J),
	    {ok,<<Lj>>} = nist:read_label(Fd2, J),
	    Mij = nist:mix_images(Mi, Mj),
	    Mv = nist:image_to_vector(Mij),
	    Y1 = deep_net:feed(Net, Mv),
	    [[Xi,Xj]] = matrix:to_list(matrix:topk(Y1, 2)),
	    if Xi-1 =:= Li, Xj-1 =:= Lj;
	       Xi-1 =:= Lj, Xj-1 =:= Li ->
		    test_mix_loop(Net,Fd1,Fd2,L-1,N+1);
	       true ->
		    test_mix_loop(Net,Fd1,Fd2,L-1,N)
	    end
    end.
