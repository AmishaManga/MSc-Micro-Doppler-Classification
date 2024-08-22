% Window = Hann(N)
%
% Generates a normalisd (area = 1) Hann window of N samples

function Window = Hann(N)

n = 0:(N-1);
Window = 0.5 - 0.5*cos(2*pi*n/(N-1));
Window = Window / sum(Window);

end