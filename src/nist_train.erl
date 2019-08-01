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



main(Hidden, Otype, Options, NTrain, NValid) ->
    {TrainingSet0,ValidationSet} = load(NTrain, NValid,
					fun nist:image_to_vector/1),
    io:format("loaded ~w+~w\n", [NTrain, NValid]),
    Net = deep_net:new(784,Hidden,{Otype,10}),
    TrainingSet = [ {X,nist:label_to_matrix(Y)} || {X,Y} <- TrainingSet0],
    deep_net:sgd(Net, TrainingSet, ValidationSet, Options).

load2d(N) ->
    load(N, fun nist:image_to_matrix/1).

loadv(N) ->
    load(N, fun nist:image_to_vector/1).

%% load one set of test data
load(N,Transform) ->
    {ok,Fd1} = nist:open_image_file(),
    {ok,Fd2} = nist:open_label_file(),
    R = load_(Fd1,Fd2,N,Transform),
    nist:close_image_file(Fd1),
    nist:close_label_file(Fd2),
    R.

%% load two sets of test data
load(N,M,Transform) ->
    {ok,Fd1} = nist:open_image_file(),
    {ok,Fd2} = nist:open_label_file(),
    Rn = load_(Fd1,Fd2,N,Transform),
    Rm = load_(Fd1,Fd2,M,Transform),
    nist:close_image_file(Fd1),
    nist:close_label_file(Fd2),
    {Rn,Rm}.

load_(_Fd1,_Fd2,0,_Transform) ->
    [];
load_(Fd1,Fd2,I,Transform) ->
    {ok,Img} = nist:read_next_image(Fd1),
    {ok,<<L>>} = nist:read_next_label(Fd2),
    [{ Transform(Img), L } | load_(Fd1,Fd2,I-1,Transform)].


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
