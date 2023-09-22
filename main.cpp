////////////////////////////////////////////////////////////////////////////////
// C++ include
#include <iostream>
#include <string>
#include <vector>
#include <limits>

// Utilities for the Assignment
#include "utils.h"
#include <gif.h>

// Image writing library
#define STB_IMAGE_WRITE_IMPLEMENTATION // Do not include this line twice in your project!
#include "stb_image_write.h"

// Shortcut to avoid Eigen:: everywhere, DO NOT USE IN .h
using namespace Eigen;

////////////////////////////////////////////////////////////////////////////////
// Scene setup, global variables
////////////////////////////////////////////////////////////////////////////////

const char* fileName ="raymarch.gif";

//Camera settings
const double focal_length = 48;
const double field_of_view = M_PI / 3; //60 degree view
const double image_z = 24;
const Vector3d camera_position(0, 0, 24);

double END = 80.0, EPSILON = 0.0001;
int MAX_STEPS = 200;

Vector3d dx(EPSILON, 0, 0);
Vector3d dy(0, EPSILON, 0);
Vector3d dz(0, 0, EPSILON);

enum ADDON{NONE, SCAR, INFECT};
enum OBJ{Clump, Wall, Back, None};

const Vector4d obj_ambient_color(0.3, 0.15, 0, 0);
const Vector4d obj_diffuse_color(0.48, 0.36, 0.288, 0);
const Vector4d obj_specular_color(0.3, 0.18, 0.144, 0);
const double obj_specular_exponent = 4.0;

const Vector4d bg_ambient_color(0.1, 0.1, 0.1, 0);
const Vector4d bg_diffuse_color(0.108, 0.027, 0.027, 0);

//Lights
std::vector<Vector3d> light_positions;
std::vector<Vector4d> light_colors;
//Ambient light
const Vector4d ambient_light(0.4, 0.2, 0.5, 1);

//Fills the different arrays
void setup_scene()
{
    //Lights
    light_positions.emplace_back(8, 8, 20);
    light_colors.emplace_back(30, 30, 30, 0);

    light_positions.emplace_back(6, -8, 11);
    light_colors.emplace_back(30, 30.53, 27.2, 0);

    light_positions.emplace_back(-6, 6, 23);
    light_colors.emplace_back(60, 60, 60, 0);

    light_positions.emplace_back(5, -10, 13);
    light_colors.emplace_back(30, 30.53, 27.2, 0);

    light_positions.emplace_back(0, -7, 20);
    light_colors.emplace_back(50, 50.2, 57.2, 0);

    light_positions.emplace_back(20, 10, 15);
    light_colors.emplace_back(30, 30, 30, 0);

    light_positions.emplace_back(-4, 8, 10);
    light_colors.emplace_back(30, 30, 30, 0);
}

// smooth min function with root smooth, instead of regular std::min
double smin(double a, double b) {
    double h = a - b;
    return 0.5 * ((a+b) - sqrt(h*h+0.01));  // k is a root smooth constant
}


////////////////////////////////////////////////////////////////////////////////
// distance functions
////////////////////////////////////////////////////////////////////////////////

// sphere in the center
double coreSDF (Vector3d &point) {
    double deform = 0.1 * sin(8 * point[0]) * sin(3 * point[1]) * sin(5 * point[2]);
    return point.norm() - 1.0 + deform;
}

// helper function to do cylinder
inline double cylinder(Vector3d &p, Vector3d a, Vector3d b, double r) {
    Vector3d pa = p - a, ba = b - a;
    double dots = pa.dot(ba) / ba.dot(ba);
    double h = std::min(std::max(dots, 0.0), 1.0);
    return (pa - h * ba).norm() - r;
}

// palm sdf: round box & capped cone intercection
double palmSDF(Vector3d &point) {
    double ra = 0.2;    
    double rb = 0.26;
    Vector3d a(3.26, 2.22, 0);
    Vector3d p = (point - a);
    Vector3d pr(0.96 * p[0] + 0.28 * p[1], -0.28*p[0]+0.96*p[1], p[2]);

    Vector3d q = pr.cwiseAbs() - Vector3d(0.35, 0.4, 0);    // round box
    Vector3d v = q.cwiseMax(0.0);
    double term = std::min(std::max(q[0], std::max(q[1], q[2])), 0.0);
    double box = v.norm() + term - 0.12;

    Vector3d b(3.52, 2.68, 0);  // capped cone
    double rba = rb - ra;
    double baba = (b-a).dot(b-a);
    double papa = (point-a).dot(point-a);
    double paba = (point-a).dot(b-a) / baba;
    double x = sqrt(papa - paba * paba * baba);
    double cax = std::max(0.0,x-((paba<0.5)?ra:rb));
    double cay = abs(paba - 0.5) - 0.5;
    double k = rba * rba + baba;
    double f = (rba * (x - ra) + paba * baba) / k;
    f = std::min(std::max(f, 0.0), 1.0);
    double cbx = x - ra - f * rba;
    double cby = paba - f;
    double s = (cbx<0.0 && cay < 0.0) ? -1.0 : 1.0;
    double cone = s * sqrt(std::min(cax * cax + cay * cay * baba, cbx * cbx + cby * cby * baba));

    // intersection of the round box and capped cone is the palm
    return std::max(box, cone); 
}

// a torus with twist, to make scar for arm
double scar(Vector3d &point) {
    Vector3d p = point - Vector3d(1.55, 1.12, 0);  // for demo arm, center of torus lies there
    p[0] = p[0] / 4;    // scale x direction by 4 so it covers full upper arm
    p[1] = 1.2 * p[1];  // scale y direction, shallower scar
    const double k = 40;    // constant for twisting
    double c = cos(k * p[0]);
    double s = sin(k * p[0]);
    Vector3d q(c*p[0]-s*p[2], s*p[0]+c*p[2], p[1]); // the twist function
    double qxz = sqrt(q[0] * q[0] + q[2] * q[2]) - 0.15;    // the torus
    Vector2d vec(qxz, q[1]);
    return vec.norm() - 0.15;
}

// a torus with displacement, to make infection on arm
double infect(Vector3d &point) {
    Vector3d p = point - Vector3d(2.89, 1.53, 0);
    p[0] = p[0] * 2.24;
    p[1] = 1.75 * p[1];
    p[2] = 2.05 * p[2];
    double noise = sin(20 * p[0]) * sin(20 * p[1]) * sin(20 * p[2]);
    double pxz = sqrt(p[0] * p[0] + p[2] * p[2]) - 0.1;
    Vector2d vec(pxz, p[1]); 
    return vec.norm() - 0.08 + noise;
}

// forearm, round cone
double foreSDF(Vector3d &point, Vector3d &b) {
    Vector3d c(3.26, 2.22, 0);
    Vector3d cb = c - b;
    double r1 = 0.29, r2 = 0.25;
    double l2 = cb.dot(cb), rr = r1 - r2;
    double a2 = l2 - rr * rr;
    double il2 = 1.0 / l2;

    Vector3d pb = point - b;
    double y = pb.dot(cb);
    double z = y - l2;
    double x2 = (l2 * pb - y * cb).norm();
    double y2 = l2 * y * y;
    double z2 = z * z * l2;

    double k = abs(rr) * rr * x2;
    double fore = 0;
    if (a2 * z2 * abs(z) / z > k)
        fore = sqrt(x2 + z2) * il2 - r2;
    else if (a2 * y2 * abs(y) / y < k)
        fore = sqrt(x2 + y2) * il2 - r1;
    else
        fore = (sqrt(x2 * a2 * il2)+y*rr) * il2 - r1;
    return fore;
}

// the demo arm
double armSDF(Vector3d &point, Vector3d &p_fore, ADDON addon) {
    // upper arm: capsule, line
    Vector3d a(0.48, 0.64, 0);
    Vector3d b(2.43, 0.66, 0);
    double upper = cylinder(point, a, b, 0.28);

    // joint: sphere
    double joint = (point - b).norm() - 0.285;

    // forearm: round cone
    double fore = foreSDF(p_fore, b);
    
    // add-on, scar or infection
    switch (addon)
    {
    case NONE:
        break;
    case SCAR:
        upper = std::max(upper, -scar(point));  // carve out the scar part
        break;
    case INFECT:
        fore = std::min(fore, infect(point));
    default:
        break;
    }

    return std::min(fore, std::min(upper, joint));
}

// the demo hand
double handSDF(Vector3d &point) {
    double hand = 0;
    double palm = palmSDF(point);

    // thumb, cylinder
    Vector3d a(3.25, 2.25, -0.05);
    Vector3d b(3.10, 2.72, 0.05);
    double r = 0.065;
    double thumb = cylinder(point, a, b, r);
    hand = std::min(thumb, palm);

    // index, cyliner
    a = Vector3d(3.23, 2.5, -0.05);
    b = Vector3d(3.435, 3.075, 0.15);
    r = 0.06;
    double index = cylinder(point, a, b, r);
    hand = std::min(index, hand);

    // middle, cylinder
    a = Vector3d(3.37, 2.55, -0.05);
    b = Vector3d(3.65, 3.10, 0.15);
    r = 0.057;
    double middle = cylinder(point, a, b, r);
    hand = std::min(middle, hand);

    // ring, cylinder
    a = Vector3d(3.5, 2.5, -0.05);
    b = Vector3d(3.825, 2.91, 0.15);
    double ring = cylinder(point, a, b, r);
    hand = std::min(hand, ring);

    // little, cylinder
    a = Vector3d(3.53, 2.4, -0.05);
    b = Vector3d(3.86, 2.65, 0.15);
    r = 0.055;
    double little = cylinder(point, a, b, r);
    hand = std::min(hand, little);

    return hand;
}

// the entire demo limb, hand + arm, with noise of twist
double limbSDF(Vector3d &point, double alpha, double beta, double gamma, ADDON addon, double phi = 0) {
    Vector3d p = Vector3d(point);
    Vector3d pa;

    if (alpha != 0) {
        Matrix3d rx;
        rx << 1, 0, 0, 
              0, cos(alpha), -sin(alpha),
              0, sin(alpha), cos(alpha);
        p = rx * p;
    }
    if (beta != 0) {
        Matrix3d ry;
        ry << cos(beta), 0, sin(beta),
              0, 1, 0,
              -sin(beta), 0, cos(beta);
        p = ry * p;
    }
    if (gamma != 0) {
        Matrix3d rz;
        rz << cos(gamma), -sin(gamma), 0,
              sin(gamma), cos(gamma), 0,
              0, 0, 1;
        p = rz * p;
    }
    if (phi != 0) {
        Vector3d b(2.43, 0.66, 0);
        Matrix3d rotation;
        rotation << cos(phi), -sin(phi), 0,
                    sin(phi), cos(phi), 0,
                    0, 0, 1;
        pa = rotation * (p - b) + b;
    } else {
        pa = p;
    }
    return std::min(handSDF(pa), armSDF(p, pa, addon));
}

// a single clump object
double singleSDF(Vector3d &point, double movement) {
    // the first demo limb
    double limb1 = limbSDF(point, 0, 0, 0, NONE, movement);
    double rv = limb1;

    // to get more limbs, rotate the demo for any degree
    double limb2 = limbSDF(point, -M_PI * 0.5, M_PI/6, M_PI * 0.2, NONE, movement);
    rv = limb2 < rv ? limb2 : rv;

    double limb3 = limbSDF(point, M_PI/3, -0.45 * M_PI, -0.75 * M_PI, SCAR);
    rv = limb3 < rv ? limb3 : rv;;

    double limb4 = limbSDF(point, M_PI * 0.3, -0.84 * M_PI, 0.36 * M_PI, INFECT);
    rv = limb4 < rv ? limb4 : rv;

    double limb6 = limbSDF(point, -0.2 * M_PI, 0.33 * M_PI, 0, INFECT);
    rv = limb6 < rv ? limb6 : rv;

    double limb7 = limbSDF(point, 0.4 * M_PI, 0, M_PI, NONE, movement);
    rv = limb7 < rv ? limb7 : rv;

    Vector3d p(-point[0], point[1], point[2]);
    double limb8 = limbSDF(p, 0, 0.3, 0, SCAR);
    rv = limb8 < rv ? limb8 : rv;

    double limb5 = limbSDF(p, -0.25 * M_PI, 0.62 * M_PI, M_PI, NONE, movement);
    rv = limb5 < rv ? limb5 : rv;

    double limb9 = limbSDF(p, 0, 0.25 * M_PI, 0.7 * M_PI, NONE, movement);
    rv = limb9 < rv ? limb9 : rv;

    double core = coreSDF(point);
    rv = smin(rv, core);
    return rv;
}

double boxSDF(Vector3d &point, Vector3d &b) {    // box for background
    
    Vector3d q = point.cwiseAbs() - b;
    Vector3d v = q.cwiseMax(0.0);
    double term = std::min(std::max(q[0], std::max(q[1], q[2])), 0.0);
    return v.norm() + term;
}

double opRep(Vector3d &point, Vector3d &c, Vector3d &b) {   // infinite repetition of the box above
    Vector3d mod0 = point + 0.5 * c;
    Vector3d flr = floor(mod0.cwiseQuotient(c).array());
    Vector3d q = mod0 - c.cwiseProduct(flr);
    return boxSDF(q, b);
}


double fullSDF(Vector3d point, OBJ &obj, double movement) {
    double rv = END;
    Vector3d p0 = point + Vector3d(-0.75, 5.23, -1.03); // 1st clump, don't move
    double clump0 = singleSDF(p0, movement);
    rv = std::min(rv, clump0);
    
    Matrix3d rot;
    double theta = M_PI / 6;
    rot << cos(theta), -sin(theta), 0,
           sin(theta), cos(theta), 0,
           0, 0, 1;
    Vector3d p1 = point + Vector3d(6.28, 1.36, -4.97);  // 2nd clump, don't move
    p1 = rot * p1;
    double clump1 = singleSDF(p1, movement);
    rv = std::min(rv, clump1);

    Vector3d p2 = point + Vector3d(-7.36, -3.55, -8.09);    // 3rd clump, move
    p2 = rot * rot * p2;
    double clump2 = singleSDF(p2, movement);
    rv = std::min(rv, clump2);

    if (rv < END) obj = OBJ::Clump;

    Vector3d b(5.1962, 3.0, END);   // pattern of background
    Vector3d c = 2.003 * b;
    Vector3d pw = point + Vector3d(-2.5981, -1.5, 4);
    double wall = opRep(pw, c, b);
    if (wall < rv) {
        rv = wall;
        obj = OBJ::Wall;
    }

    pw = pw + Vector3d(0, 0, 6);
    c = Vector3d(30, 30, 0.1);
    double back = boxSDF(pw, c);
    if (back < rv) {
        rv = back;
        obj = OBJ::Back;
    }
    return rv;    
}


//We need to make this function visible
Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, double phi);


////////////////////////////////////////////////////////////////////////////////
// Raymarching code
////////////////////////////////////////////////////////////////////////////////

double marching(const Vector3d &ray_origin, const Vector3d &ray_direction, OBJ &obj, double phi) {
    double depth = 0.5; // beginning value of distance, akin to t in ray tracing 
    for (int i = 0; i < MAX_STEPS; i++) {
        Vector3d point = ray_origin + depth * ray_direction;
        double dist = fullSDF(point, obj, phi);
        if (dist < EPSILON)
            return depth;
        depth += dist;
        if (depth >= END) {
            return END;
        }
    }
    return END;
}

double shadow(const Vector3d &ray_origin, const Vector3d &ray_direction, double k, double phi) {
    double res = 1.0;
    double t = 0.5;
    OBJ dump;
    for (int i = 0; i < MAX_STEPS && t<15.0; i++) {
        Vector3d point = ray_origin + t* ray_direction;
        double h = fullSDF(point, dump, phi);
        if (h < EPSILON) return 0.0;
        res = std::min(res, k*h/t);
        t += h;
    }
    return res;
}

Vector4d shoot_ray(const Vector3d &ray_origin, const Vector3d &ray_direction, double phi) {
    OBJ obj, ndump;

    double t = marching(ray_origin, ray_direction, obj, phi);
    double s = shadow(ray_origin, ray_direction, 8.0, phi);
    //printf("%3f\n", s);
    if (t == END) return Vector4d(0, 0, 0, 1);

    Vector4d ambient_color; //= obj_ambient_color.array() * ambient_light.array();
    // intersection point
    const Vector3d p = ray_origin + t * ray_direction;
    // little steps to estimate normal
    
    // normal, estimated
    Vector3d N(fullSDF(p + dx, ndump, phi) - fullSDF(p - dx, ndump, phi), 
        fullSDF(p + dy, ndump, phi) - fullSDF(p - dy, ndump, phi), fullSDF(p + dz, ndump, phi) - fullSDF(p - dz, ndump, phi));
    N = N.normalized();

    // color from the lights
    Vector4d lights_color(0, 0, 0, 0);
    for (int i = 0; i < light_positions.size(); i++) {
        const Vector3d &light_pos = light_positions[i];
        const Vector4d &light_color = light_colors[i];
        const Vector3d Li = (light_pos - p).normalized();
        // diffuse
        Vector4d diff_col; // = obj == OBJ::Clump ? obj_diffuse_color : bg_diffuse_color;
        switch (obj)
        {
        case Clump:
            diff_col = obj_diffuse_color;
            ambient_color = obj_ambient_color.cwiseProduct(ambient_light);
            break;
        case Wall:
            diff_col = bg_diffuse_color;
            ambient_color = obj_ambient_color.cwiseProduct(ambient_light);
            break;
        default:
            diff_col = 0.48 * bg_diffuse_color + 0.06 * obj_diffuse_color;
            ambient_color = bg_ambient_color.cwiseProduct(ambient_light);
            break;
        }
        Vector4d diffuse = std::max(Li.dot(N), 0.0) * diff_col;
        // specular
        Vector4d specular;
        if (obj != OBJ::Clump) {
            specular << 0, 0, 0, 0;
        } else {
            Vector3d v = (camera_position - p).normalized();
            double vh = (v + Li).normalized().transpose().dot(N);
            vh = std::pow(vh, obj_specular_exponent);
            specular = std::max(vh, 0.0) * obj_specular_color;
        }

        // Attenuate lights according to the squared distance to the lights
        const Vector3d D = light_pos - p;
        lights_color += (diffuse + specular).cwiseProduct(light_color) / D.squaredNorm();

    }
    Vector4d C = ambient_color + lights_color;
    if (obj != OBJ::Clump) C = C * s;
    C(3) = 1;
    return C;
}

////////////////////////////////////////////////////////////////////////////////

void raytrace_scene()
{
    std::cout << "Simple ray marching." << std::endl;

    int w = 800;
    int h = 600;
    MatrixXd R = MatrixXd::Zero(w, h);
    MatrixXd G = MatrixXd::Zero(w, h);
    MatrixXd B = MatrixXd::Zero(w, h);
    MatrixXd A = MatrixXd::Zero(w, h); // Store the alpha mask

    // The camera always points in the direction -z
    // The sensor grid is at a distance 'focal_length' from the camera center,
    // and covers an viewing angle given by 'field_of_view'.
    double aspect_ratio = double(w) / double(h);
    double image_y = focal_length * tan(field_of_view * 0.5); //TODO: compute the correct pixels size
    double image_x = image_y * aspect_ratio; //1; //TODO: compute the correct pixels size

    // The pixel grid through which we shoot rays is at a distance 'focal_length'
    const Vector3d image_origin(-image_x, image_y, -image_z);
    const Vector3d x_displacement(2.0 / w * image_x, 0, 0);
    const Vector3d y_displacement(0, -2.0 / h * image_y, 0);

    std::vector<uint8_t> image;
    image.resize(w * h * 4, 0);
	int delay = 25;
	GifWriter g;
	GifBegin(&g, fileName, w, h, delay);

    for (int k = 0; k < 10; k++) {
        printf("doing frame %d\n", k);
        double theta = k * 0.0125 * M_PI;
        for (unsigned i = 0; i < w; ++i)
        {
            if (i % 80 == 0) printf("   loading %2d precent\n", i * 100 / w);
            for (unsigned j = 0; j < h; ++j)
            {
                const Vector3d pixel_center = image_origin +(i+0.5) * x_displacement + (j+0.5) * y_displacement;

                // Prepare the ray
                Vector3d ray_origin;
                Vector3d ray_direction;

                ray_origin = camera_position;
                ray_direction = (pixel_center - ray_origin).normalized();
                const Vector4d C = shoot_ray(ray_origin, ray_direction, theta);

                image[(j * w * 4) + (i * 4) + 0] = (uint8_t)roundf( 255.0f * C(0));
                image[(j * w * 4) + (i * 4) + 1] = (uint8_t)roundf( 255.0f * C(1));
                image[(j * w * 4) + (i * 4) + 2] = (uint8_t)roundf( 255.0f * C(2));
                image[(j * w * 4) + (i * 4) + 3] = 255;
            }
        }
        GifWriteFrame(&g, image.data(), w, h, delay);
    }

    for (int k = 10; k > 0; k--) {
        printf("doing frame %d\n", 20-k);
        double theta = k * 0.0125 * M_PI;
        for (unsigned i = 0; i < w; ++i)
        {
            if (i % 80 == 0) printf("   loading %2d precent\n", i * 100 / w);
            for (unsigned j = 0; j < h; ++j)
            {
                const Vector3d pixel_center = image_origin +(i+0.5) * x_displacement + (j+0.5) * y_displacement;

                // Prepare the ray
                Vector3d ray_origin;
                Vector3d ray_direction;

                ray_origin = camera_position;
                ray_direction = (pixel_center - ray_origin).normalized();
                const Vector4d C = shoot_ray(ray_origin, ray_direction, theta);
    
                image[(j * w * 4) + (i * 4) + 0] = (uint8_t)roundf( 255.0f * C(0));
                image[(j * w * 4) + (i * 4) + 1] = (uint8_t)roundf( 255.0f * C(1));
                image[(j * w * 4) + (i * 4) + 2] = (uint8_t)roundf( 255.0f * C(2));
                image[(j * w * 4) + (i * 4) + 3] = 255;
            }
        }
        GifWriteFrame(&g, image.data(), w, h, delay);
    }
    GifEnd(&g);
}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
    setup_scene();

    raytrace_scene();
    return 0;
}
