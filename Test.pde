import processing.video.*;


int cx, cy;
int num = 6;
PVector[] inkArray = new PVector[num];


PVector prevPos = new PVector(0,0);
PVector currPos = new PVector(10,10);

float[] maxSize = new float[num];

PImage pim;
float a ,w ;
float row_time = 500;
float curr_time = 0;
int curr_row = 0;


PVector[] p , speed;
int nb = 3, imageW, imageH;
float c1 = 4, c2 = 2;//noise coefficients, the higher the more strata// 40, 2

ink_flow test_flow;
ink_scan fluid;


//live video feed

Capture cam;

void setup()
{
  background(255);
  size(400,400,P3D);
  pim = loadImage("Data/winston.jpg");
  cx = width/2;
  cy = height/2;
  
  for(int i=0; i<num; i++) {
    inkArray[i] = new PVector(random(-100, 100) + cx, 0);
    
    maxSize[i] = random(height);
  }
  //stroke(pim.get((int)inkArray[curr_row].x,(int) inkArray[curr_row].y),a);
  pim.resize(400,400);
  image(pim,0,0);
  //println(pim.pixels.length);
  imageW = pim.width;
  imageH = pim.height;
  p = new PVector[nb];
  speed = new PVector[nb];
  initNoise();
  
  //test_flow = new ink_flow();
 // test_flow = new ink_flow(pim.get(250,10),250,10);
  //println(pim.get(width/2,height/2));
 // test_flow = new ink_flow(color(255,255,0),50,0);
 
 
 
 String[] cameras = Capture.list();
 cam = new Capture(this,cameras[0]);
 cam.start();
 fluid = new ink_scan(400,pim);
}
void initNoise()
{
  for (int i = 0; i < nb; i ++)
  {
    p[i] = new PVector(random(123), random(123)); 
    speed[i] = new PVector(random(-.02*(i+1), .02*(i+1)), random(-.02*(i+1), .02*(i+1)));//0, 0);//
  }
  
}

void init(int curr_row)
{
  for(int i=0; i<num; i++) {
    inkArray[i] = new PVector(random(-100, 100) + cx, curr_row);
    maxSize[i] = random(height);
  }
  stroke(pim.get((int)inkArray[curr_row].x,(int) inkArray[curr_row].y),a);
  
}
/*
void draw()
{
  test_flow.update();
  test_flow.draw_flow();
}
*/
void draw() 
{
  
 /*
  //liquid image
  pim.loadPixels();
  loadPixels();
  for (float x = 0; x < width; x++) {
    for (float y = 0; y < height; y++) {
      pixels[int(x)+int(y)*width] = color(pattern(x, y),50);
    }
  }
  updatePixels();
  pim.updatePixels();

  for (int i = 0; i < nb; i ++)
  {
    p[i].add(speed[i]);
  }
*/
  //fill(255,0,0,255);
  //rect(0,0,width,height);
 
 // test_flow.update();
  //test_flow.draw_flow();
 // cam.read();
  //image(cam,0,0);
  fluid.update();
  fluid.draw_flow();
  /*
  cam.loadPixels();
  for (float x = 0; x < cam.width; x++) {
    for (float y = 0; y < cam.height; y++) {
      cam.pixels[int(x)+int(y)*width] = color(pattern(x, y),50);
    }
  }
  cam.updatePixels();
  */
  
  /*
  for(int i=0; i<num; i++) 
  {
    prevPos.x = inkArray[i].x;
    prevPos.y = inkArray[i].y;
    inkArray[i].x += random(-1.5, 1.5);
    inkArray[i].y += random(1);
    w = abs(maxSize[i] / (inkArray[i].y + 25));
    a = map(inkArray[i].y, maxSize[i], 0, 0, 200);
    strokeWeight(w);
    //stroke(255,0,0, a);
    //stroke(pim.get((int)inkArray[i].x,(int) inkArray[i].y),a);
    if (inkArray[i].y < maxSize[i]) {
      line(prevPos.x, prevPos.y, inkArray[i].x, inkArray[i].y);
    } else {
      inkArray[i] = new PVector(random(-100, 100) + cx, 0); 
      maxSize[i] = random(height);
    }
  }
  if(curr_time >= row_time)
  {
    curr_time = 0;
    if(curr_row < height)
    {
      curr_row+=10;
      init(curr_row);
    }
  }
  curr_time++;
  
 */ 
  
  
  
  
}

int pattern(float px, float py)
{
  float qx, qy, rx, ry, w, r, g, b;
  int myColor = 0;
  int x = int(px), y = int(py);

  px /= 100;
  py /= 100;

  qx = noise(px + p[0].x, py + p[0].y);//regular noise
  qy = noise(px + p[0].x + .52, py + p[0].y + .13);
  rx = noise(px + p[1].x + c1*qx + .17, py + p[1].y + c1*qy + .92);
  ry = noise(px + p[1].x + c1*qx + .83, py + p[1].y + c1*qy + .28);
  w = noise(px + p[2].x + c2*rx, py + p[2].y + c2*ry);

  //x = constrain(int(x + map(mouseX, 0, width, 0, 50) * ((new PVector(rx, ry)).mag()-1)), 0, imageW-1);
  //y = constrain(int(y + map(mouseY, 0, height, 0, 50) * ((new PVector(qx, qy)).mag()-1)), 0, imageH-1);
 

  x = constrain(int(x +2 * ((new PVector(rx, ry)).mag()-1)), 0, imageW-1);
  y = constrain(int(y + 2 * ((new PVector(qx, qy)).mag()-1)), 0, imageH-1);
  
  myColor = pim.pixels[x + y * imageW];
  return myColor;
}

void mousePressed()
{
  initNoise();
}
