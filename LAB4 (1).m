disp('Adaline n/w For OR Function bipolar Inputs and Target ');
i1=[1 1 -1 -1];
i2=[1 -1 1 -1];
i3=[1 1 1 1];
t=[1 1 1 -1];
w1=0.2;
w2=0.2;
alpha=0.2;
b=0.2;
delw1=0;
delw2=0;
delb=0;
epoch=0;
e=0;
while(e<0.5)
    epoch=epoch+1;
    e=0;
    for j=1:4
        finaly(j)=w1*i1(j)+w2*i2(j)+b;
        nt=[finaly(j) t(j)];
        delw1=alpha*(t(j)-finaly(j)*i1(j));
        delw2=alpha*(t(j)-finaly(j)*i2(j));
        delb=alpha*(t(j)-finaly(j))*i3(j);
        wc=[delw1 delw2 delb];
        w1=w1+delw1;
        w2=w2+delw2;
        b=b+delb;
        w=[w1 w2 b];
        i=[i1(j) i2(j) i3(j)];
        out=[i nt wc w];
    end
    for k=1:4
        finaly(k)=w1*i1(k)+w2*i2(k)+k;
        e=e+(t(k)-finaly(k))^2;
    end
    if epoch==1
    end
end
figure
hold on
for i=1:4
    nety(i)=w1*i1(i)+w2*i2(i)+b;
    e=e+(t(i)-nety(i))^2;
    stem(nety(i),i);
    legend boxon
    grid on
end



