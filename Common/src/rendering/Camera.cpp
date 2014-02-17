#include "rendering/Camera.h"

#include <glm/gtx/projection.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace vlr
{
	namespace common
	{
		Camera::Camera()
		{

		}
		
		Camera::Camera(const glm::vec3& position, const glm::quat& rotation)
			: _position(position), _rotation(rotation)
		{
			setViewport(0, 0, 0, 0);
		}

		Camera::~Camera()
		{

		}

		void Camera::updateGL()
		{
			// Update view matrix
			_view = glm::toMat4(_rotation);
			_view = glm::translate(_view, -_position);

			// Calculate mvp
			glm::mat4 mvp = getMVP();

			// Set viewport
			glViewport(_viewport.x, _viewport.y, _viewport.w, _viewport.h);

			// Set projection
			glMatrixMode(GL_PROJECTION);
			glLoadMatrixf(glm::value_ptr(mvp));

			// Initialise modelview
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
		}

		void Camera::setViewport(int x, int y, int width, int height)
		{
			_viewport.x = x;
			_viewport.y = y;
			_viewport.w = width;
			_viewport.h = height;
		}

		void Camera::perspective(float fov, float aspect,
			float near, float far)
		{
			_projection = glm::perspective(fov, aspect, near, far);
		}

		void Camera::orthographic(float scale, float aspect)
		{
			_projection = glm::ortho(scale * -1.0f, scale * 1.0f,
				(scale * -1.0f) / aspect, (scale * 1.0f) / aspect);
		}


		void Camera::setPos(const glm::vec3& pos)
		{
			_position = pos;
		}

		void Camera::setRot(const glm::quat quat)
		{
			_rotation = quat;
		}

		void Camera::translate(const glm::vec3& offset)
		{
			_position += offset;
		}

		void Camera::rotate(const glm::quat& quat)
		{
			_rotation *= quat;
		}

		void Camera::rotate(const glm::vec3& euler)
		{
			_rotation *= glm::quat(euler);
		}

		glm::vec3 Camera::screenSpaceToWorld(float x, float y,
			float dist) const
		{
			// Convert x and y to viewport space
			x = (x / _viewport.w) * 2.0f - 1.0f;
			y = (y / _viewport.h) * 2.0f - 1.0f;

			glm::vec3 point(x, y, dist);

			glm::vec3 unprojected = glm::unProject(point, _view * _model,
				_projection, glm::vec4(_viewport.x, _viewport.y, _viewport.w,
				_viewport.h));

			return unprojected;
		}
	}
}
