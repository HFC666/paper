x = -pi:pi/2:pi;
y = sin(x);
new_x = -pi:0.1:pi;
p = pchip(x, y, new_x);
plot(x, y, '*r', new_x, p, '.b-')