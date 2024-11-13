I_base = uint8(zeros([64,64]))+255;

% Create a deck of seed images of hearts
I_deck = []
for i=1:9
    s = strcat('gen_images_test\hearts\h',int2str(i),'.png');
    I1 = imread(s);
    I1 = im2gray(I1);
    I1 = imresize(I1,[64 64]);
    I1 = uint8(I1 <= 240).*I1 + uint8(I1 > 240) .* 255;
    I_deck = cat(3, I_deck,I1);
    subplot(3,3,i)
    imshow(I1)
end


% % Create different images with a heart present.
% for i=1:9
%     rszx = randi([30,50],1);
%     rszy = randi([30,50],1);
%     randpic = randi([1,9],1);
%     randrot = randi([-180,180],1);
%     randlocx = randi([1, 64-rszx],1);
%     randlocy = randi([1, 64-rszy],1);
%     I_rand = I_deck(:,:,randpic);
%     I_rand = imrotate_white(I_rand,randrot);
%     I_rsz = imresize(I_rand,[rszx rszy]);
% 
%     new_I = I_base;
%     new_I(randlocx:(randlocx+rszx-1),randlocy:(randlocy+rszy-1)) = I_rsz;
% 
%     subplot(3,3,i)
%     imshow(new_I)
% end


%%

for i=1:9
    s = strcat('gen_images_test\diamonds\d',int2str(i),'.png');
    I1 = imread(s);
    I1 = im2gray(I1);
    I1 = imresize(I1,[64 64]);
    I1 = uint8(I1 <= 240).*I1 + uint8(I1 > 240) .* 255;
    I_deck = cat(3, I_deck,I1);
end

for i=1:9
    s = strcat('gen_images_test\spades\s',int2str(i),'.png');
    I1 = imread(s);
    I1 = im2gray(I1);
    I1 = imresize(I1,[64 64]);
    I1 = uint8(I1 <= 240).*I1 + uint8(I1 > 240) .* 255;
    I_deck = cat(3, I_deck,I1);
    subplot(3,3,i)
    imshow(I1)
end

for i=1:9
    s = strcat('gen_images_test\clubs\c',int2str(i),'.png');
    I1 = imread(s);
    I1 = im2gray(I1);
    I1 = imresize(I1,[64 64]);
    I1 = uint8(I1 <= 240).*I1 + uint8(I1 > 240) .* 255;
    I_deck = cat(3, I_deck,I1);
    subplot(3,3,i)
    imshow(I1)
end

yhlabel = 000;
yslabel = 000;
yclabel = 000;
ydlabel = 000;

nhlabel = 000;
nslabel = 000;
nclabel = 000;
ndlabel = 000;
% Create different images with up to five icons present.
for i=1:80000
    rn = randi([1,5],1);
    new_I = I_base;
    icons = [];
    for j=1:rn
        overlap = 1;
        trying = 0;
        while overlap && trying < 1000
            rszx = randi([15,50],1);
            rszy = randi([15,50],1);
            randpic = randi([1,36],1);
            randrot = randi([-180,180],1);
            randlocx = randi([1, 64-rszx],1);
            randlocy = randi([1, 64-rszy],1);
            I_rand = I_deck(:,:,randpic);
            I_rand = imrotate_white(I_rand,randrot);
            I_rsz = imresize(I_rand,[rszx rszy]);
    
            overlap = ~( sum(sum( new_I(randlocx:(randlocx+rszx-1),randlocy:(randlocy+rszy-1)) ~= 255 )) == 0);
        
            if ~overlap
                new_I(randlocx:(randlocx+rszx-1),randlocy:(randlocy+rszy-1)) = I_rsz;
            end
        trying = trying+1;
        end
        if trying < 1000
            icons = cat(1, icons, randpic);
        end
    end

  
    if sum( icons <= 9 ) >= 1
        % Then a heart is present
        s = strcat('hearts\',int2str(yhlabel),'.png');
        imwrite(new_I, s);
        yhlabel = yhlabel +1;
        if yhlabel > 9999
            break
        end 
    else
        % Then there's no heart
        s = strcat('nohearts\',int2str(nhlabel),'.png');
        imwrite(new_I, s);
        nhlabel = nhlabel +1;
        if nhlabel > 9999
            break
        end
      
    end

    if sum( (icons > 9).*(icons <=18) ) >= 1
        % Then a diamond is present
        s = strcat('diamonds\',int2str(ydlabel),'.png');
        imwrite(new_I, s);
        ydlabel = ydlabel +1;
        if ydlabel > 9999
            break
        end
    else
        % Then there's no diamond
        s = strcat('nodiamonds\',int2str(ndlabel),'.png');
        imwrite(new_I, s);
        ndlabel = ndlabel +1;
        if ndlabel > 9999
            break
        end
    end

    if sum( (icons > 18).*(icons <=27) ) >= 1
        % Then a spade is present
        s = strcat('spades\',int2str(yslabel),'.png');
        imwrite(new_I, s);
        yslabel = yslabel +1;
        if yslabel > 9999
            break
        end
    else
        % Then there's no spade
        s = strcat('nospades\',int2str(nslabel),'.png');
        imwrite(new_I, s);
        nslabel = nslabel +1;
        if nslabel > 9999
            break
        end
    end

    if sum( (icons > 27)) >= 1
        % Then a club is present
        s = strcat('clubs\',int2str(yclabel),'.png');
        imwrite(new_I, s);
        yclabel = yclabel +1;
        if yclabel > 9999
            break
        end
    else
        % Then there's no club
        s = strcat('noclubs\',int2str(nclabel),'.png');
        imwrite(new_I, s);
        nclabel = nclabel +1;
        if nclabel > 9999
            break
        end
    end
end
