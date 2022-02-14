#include <iostream>
#include <sstream>
#include <string>
#include <fstream>
#include <opencv2/highgui.hpp>


struct Calibration_param
{
    /* data */
    cv::Mat K0;
    cv::Mat K1;
    int doffs;
    float baseline;
    int witdh;
    int height;
    int ndisp;
    int vmin;
    int vmax;

    cv::Mat parseK(std::istringstream& iss){
        std::string a,b,c, d,e,f, g,h,i;
                    
        iss >> a >> b >> c >> d >> e >> f >> g >> h >> i ;

        a.erase(a.begin()); // remove '['
        c.pop_back(); // remove ';'
        f.pop_back(); // remove ';'
        i.pop_back(); // remove ']'
        
        double m[3][3] = {{std::stod(a), std::stod(b), std::stod(c)},
                            {std::stod(d), std::stod(e), std::stod(f)},
                            {std::stod(g), std::stod(h), std::stod(i)}};
        cv::Mat M = cv::Mat(3, 3, CV_64F, m);
        
        return M;
    }

    friend std::istream& operator>>(std::istream& file, Calibration_param& data)
    {
    std::string line, name;
    std::istringstream iss;

        while (std::getline(file, line))
        {
            std::istringstream iss(line);

            if(std::getline(iss, name, '=')){
                if (name=="cam0"){
                    data.K0 = data.parseK(iss);
                }else if (name=="cam1"){
                    data.K1 = data.parseK(iss);
                }else if (name=="doffs"){
                    iss >> data.doffs;
                }else if (name=="baseline"){
                    iss >> data.baseline;
                }else if (name=="width"){
                    iss >> data.witdh;
                }else if (name=="height"){
                    iss >> data.height;
                }else if (name=="ndisp"){
                    iss >> data.ndisp;
                }else if (name=="vmin"){
                    iss >> data.vmin;
                }else if (name=="vmax"){
                    iss >> data.vmax;
                }
            }
            

        }
        
        return file;
    }
    
};


int main() {
    std::string calibFilePath = "../datas/all/data/artroom1/calib.txt";
    std::ifstream f(calibFilePath);
    std::string line, name;
    std::istringstream iss;

    Calibration_param data;
    f >> data;
    std::cout << data.K0 << "\n";
    std::cout << data.K1 << "\n";
    std::cout << data.doffs << "\n";
    std::cout << data.baseline << "\n";
    std::cout << data.witdh << "\n";
    std::cout << data.height << "\n";
    std::cout << data.ndisp << "\n";
    std::cout << data.vmin << "\n";
    std::cout << data.vmax << "\n";

    return 0;
}