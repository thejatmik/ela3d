clear all
x=linspace(0,pi,201);
y=linspace(0,pi,201);
[X Y]=meshgrid(x);
Z=(((sin(X)+sin(Y))/2)-0.5)/0.5;

for i=1:201
    for j=1:201
        if (Z(i,j)<0)
            Z(i,j)=0;
        end
    end
end

surface=Z;

h3=201;
d3=100;
h2=161;
d2=100;
h1=121;
d1=100;

surface1=surface.*d1;
surface2=surface.*d2;
surface3=surface.*d3;

surface1=h1-surface1;
surface2=h2-surface2;
surface3=h3-surface3;

%surf(surface1); hold on;
%surf(surface2); hold on;
%surf(surface3); hold on;
%shading interp
%zlim([0 200]);
%ylim([0 200]);
%xlim([0 200]);

model=zeros(201*201*201,1);

for k=1:201
    for j=1:201
        for i=1:201
            ijk=i+(j-1)*201+(k-1)*201*201;
            model(ijk)=1;
            if (k>surface1(i,j))
                model(ijk)=2;
            end
            if (k>surface2(i,j))
                model(ijk)=3;
                if (k<80)
                    model(ijk)=4;
                end
            end
            if (k>surface3(i,j))
                model(ijk)=5;
            end
        end
    end
end

reshape(model,[201 201 201]);
slice(ans, 100, 100,100)
shading interp

modcp=zeros(201*201*201,1);
modcs=zeros(201*201*201,1);
modrh=zeros(201*201*201,1);

%        vp   vs   rho
medium1=[3000 1700 2800]; %usual = sandstone = 1
medium2=[3600 1400 2300]; %caprock = shale = 2
medium3=[3000 1700 2800]; %sandstone = 3
medium4=[3600 1700 2800]; %oil in sandstone = 4
medium5=[3650 1500 2450]; %source rock = shale = 5

for k=1:201
    for j=1:201
        for i=1:201
            ijk=i+(j-1)*201+(k-1)*201*201;
            if (model(ijk)==1)
                modcp(ijk)=medium1(1);
                modcs(ijk)=medium1(2);
                modrh(ijk)=medium1(3);
            end
            if (model(ijk)==2)
                modcp(ijk)=medium2(1);
                modcs(ijk)=medium2(2);
                modrh(ijk)=medium2(3);
            end
            if (model(ijk)==3)
                modcp(ijk)=medium3(1);
                modcs(ijk)=medium3(2);
                modrh(ijk)=medium3(3);
            end
            if (model(ijk)==4)
                modcp(ijk)=medium4(1);
                modcs(ijk)=medium4(2);
                modrh(ijk)=medium4(3);
            end
            if (model(ijk)==5)
                modcp(ijk)=medium5(1);
                modcs(ijk)=medium5(2);
                modrh(ijk)=medium5(3);
            end
        end
    end
end

