//try for the mass to accumulate;

class ink_flow
{
  color c;
  ArrayList<PVector>flow;
  boolean can_draw = true;
  float max_ink_size;
  ink_flow()
  {
    c = color(255,0,0);
    PVector init_pt = new PVector(width/2, 0);
    max_ink_size = random(height,2*height);
    flow = new ArrayList<PVector>();
    flow.add(init_pt);
    
  }
  ink_flow(color _c, float x, float y)
  {
    c = _c;
    //PVector init_pt = new PVector(width/2, 0);
    PVector init_pt = new PVector(x,y);
    max_ink_size = random(height,2*height);
    flow = new ArrayList<PVector>();
    flow.add(init_pt);
  }
  void update()
  {
    if(flow.get(flow.size()-1).y < height)
    {
      can_draw=true;
      PVector new_pt = new PVector(flow.get(flow.size()-1).x+random(-0.5,0.5),flow.get(flow.size()-1).y+random(1));
      flow.add(new_pt);
    }
    else
    {
      //print("False now");
      can_draw = false;
    }
  }
  void draw_flow()
  {
    
      for(int i = 0 ; i < flow.size()-1 ; i++)
      {
        //stroke(c, map(flow.get(flow.size()-1).y,max_ink_size,0,0,200));
    
        //strokeWeight(abs(max_ink_size/(flow.get(flow.size()-1).y+25 )));
        //line(flow.get(flow.size()-2).x,flow.get(flow.size()-2).y,flow.get(flow.size()-1).x,flow.get(flow.size()-1).y);
      
      stroke(c, map(flow.get(i).y,max_ink_size,0,0,200));
    
      strokeWeight(abs(max_ink_size/(flow.get(i).y+25 )));
      
       line(flow.get(i).x,flow.get(i).y,flow.get(i+1).x,flow.get(i+1).y);
       //line(flow.get(i).x,flow.get(i).y,_x,_y);
      
    } 
  }
  
}
