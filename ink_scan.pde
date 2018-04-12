class ink_scan
{
  ArrayList<ink_flow>all_ink;
  int window = 1; // number of rows that we are dealing with
  int stride = 5; // how much are we shifting
  
  int curr_row = 0;
  
  int num_flows;
  PImage pim;
  int time_to_update = 50;
  int curr_time = 0;
  ink_scan(int img_width, PImage img)
  {
    all_ink = new ArrayList<ink_flow>();
     num_flows = (int)random(10);
     num_flows = 10;
     pim = img;
    for(int i = 0 ; i < num_flows; i++)
    {
      int x_pos = (int)random(img_width);
      ink_flow curr_flow = new ink_flow(pim.get(x_pos,curr_row),x_pos,curr_row);
      all_ink.add(curr_flow);
    }
    
  }
  
  void update()
  {
    for(int i = 0 ; i < all_ink.size(); i++)
      all_ink.get(i).update();
      
      if(curr_time >= time_to_update)
      {
        curr_time = 0;
        add_scan_line();
      }
      curr_time++;
  }
  
  void draw_flow()
  {
    for(int i = 0 ; i < all_ink.size(); i++)
      all_ink.get(i).draw_flow();
  }
  
  void add_scan_line()
  {
    
    curr_row+=stride;
    if(curr_row < height)
    {
     num_flows = (int)random(10);
    for(int i = 0 ; i < num_flows; i++)
    {
      int x_pos = (int)random(pim.width);
      ink_flow curr_flow = new ink_flow(pim.get(x_pos,curr_row),x_pos,curr_row);
      all_ink.add(curr_flow);
    }
    }
  }
}
