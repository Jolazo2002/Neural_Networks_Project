x=[0 0;1 0;0 1;1 1];
y=[0;1;1;0];

ynew=onehotencode(y,2,"ClassNames",[0;1]);

inputs=2;
outputs=2;

w_num1=2;
w_num2=outputs;


weights=cell(1,2);

weights{1,1}=normrnd(0,1,[inputs w_num1])*sqrt(2/inputs);
weights{1,2}=normrnd(0,1,[w_num1 w_num2])*sqrt(2/w_num1);

biases=cell(1,2);

biases{1,1}=zeros(w_num1,1);
biases{1,2}=zeros(w_num2,1);


plotdata=zeros(1,100000);

for i=1:100000
    loss1=zeros(1,4);
    loss2=zeros(1,4);

    for j=1:4
        [loss1(j),b,c]=backprobagation(weights,biases,transpose(x(j,:)),transpose(ynew(j,:)),0.01);
        weights=b;
        biases=c;
        output=forwardcalc(weights,biases,transpose(x(j,:)));
        loss2(j)=(1/2)*sum((output{end}-transpose(ynew(j,:))).^2);
    end

    n=mean(loss1);
    m=mean(loss2);
    plotdata(i)=m;

    disp(['Run #',num2str(i),' Error Before:',num2str(n),' Error After:',num2str(n)]);
    

   %{ 
    [a,b,c]=backprobagation(weights,biases,transpose(x(1,:)),y(1),0.01);
    weights=b;
    biases=c;
    output=forwardcalc(weights,biases,transpose(x(1,:)));
    lossafter=(1/2)*sum((output{end}-y(1)).^2);
    disp(['Run #',num2str(i),' Error Before:',num2str(a),' Error After:',num2str(lossafter)]); 
   %}

end

plot(1:100000,plotdata);

output=forwardcalc(weights,biases,transpose(x(3,:)));
disp(num2str(output{end}));

save weightsb.mat weights biases;

function a=Relu(input)
    a=max(input,0);
end



function u=forwardcalc(weights,biases,input)  
    u=cell(1,2*length(weights)+1);
    u{1,1}=input;
    for i=1:length(weights)
        k=2*i;
        u{k} = transpose(weights{1,i}) * u{k-1} + biases{1,i};
        u{k+1}=Relu(u{k});
    end
end

function [lossbefore,weightsnew,biasesnew]=backprobagation(weights,biases,input,desired,learning_rate)
    output=forwardcalc(weights,biases,input);
    k=length(weights);

    lossbefore=(1/2)*sum((output{end}-desired).^2);


    delta=cell(1,k);
    dw=cell(1,k);
    dbiases=cell(1,k);

    delta{k}=(output{2*k+1}-desired).*(Relu(output{2*k+1})./output{2*k+1});
    dw{k}=output{2*k-1}*transpose(delta{k});
    dbiases{k}=delta{k};
    weightsnew{k}=weights{k}-dw{k}*learning_rate;
    biasesnew{k}=biases{k}-dbiases{k}*learning_rate;

   for i=k-1:-1:1
       delta{i}=delta{i+1}.*(Relu(output{2*k+1})./output{2*k+1});
       dw{i}=output{2*i-1}*transpose(delta{i});
       dbiases{i}=delta{i};
       weightsnew{i}=weights{i}-dw{i}*learning_rate;
       biasesnew{i}=biases{i}-dbiases{i}*learning_rate;
       
   end

   
end